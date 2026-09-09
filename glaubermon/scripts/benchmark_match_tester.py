"""Automated Tournament & Benchmark Match Tester for Glaubermon Max.

Benchmarks Glaubermon Max against multiple opponent archetypes:
1. Greedy Damage Maximizer (clicks maximum damage move)
2. Tactical Heuristic Player (evaluates hazards, stat boosts, type matchups)
3. Previous Checkpoint (AlphaZero vs Pure Grandmaster Behavior Cloning)
"""

import os
import time
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import torch
import numpy as np

from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import ActionType, PokemonType, Hazard, MoveCategory
from glaubermon.core.constants import get_type_effectiveness
from glaubermon.core.actions import Action
from glaubermon.models.set_transformer import GlaubermonMaxNet
from glaubermon.search.evaluators import NeuralEvaluator, HeuristicEvaluator, HybridEvaluator
from glaubermon.search.subgame_resolver import SubgameResolver, simulate_turn_transition
from glaubermon.inference.damage_calc import calculate_damage_rolls
from glaubermon.scripts.train_alphazero import generate_competitive_battle
from glaubermon.data.meta_teams import get_meta_team_balance


class PokeEnvMaxBasePowerPlayer:
    """Exact implementation of poke-env's MaxBasePowerPlayer (Haris Sahovic, IEEE CoG).
    
    Picks the move with highest: base_power * (1.5 if STAB) * type_effectiveness.
    """

    def select_action(self, state: BattleState, player: int = 2) -> Action:
        side = state.p2 if player == 2 else state.p1
        opp_side = state.p1 if player == 2 else state.p2
        active = side.active_pokemon
        opp_active = opp_side.active_pokemon

        valid_actions = state.get_valid_actions(player=player)
        move_actions = [a for a in valid_actions if a.action_type == ActionType.MOVE]
        if not move_actions:
            return valid_actions[0]

        best_act = move_actions[0]
        best_score = -1.0

        for ma in move_actions:
            slot = getattr(ma, "move_slot", 1) - 1
            if active and 0 <= slot < len(active.moves):
                mv = active.moves[slot]
                bp = mv.base_power
                stab = 1.5 if mv.move_type in active.active_types else 1.0
                eff = 1.0
                if opp_active:
                    eff = get_type_effectiveness(mv.move_type, opp_active.active_types[0], opp_active.active_types[1])
                score = bp * stab * eff
                if score > best_score:
                    best_score = score
                    best_act = ma

        return best_act


class PokeEnvSimpleHeuristicsPlayer:
    """Exact implementation of poke-env's SimpleHeuristicsPlayer (Haris Sahovic, IEEE CoG).
    
    Full handcrafted expert system from the official poke-env repository:
    - Calculates matchup differences: max(opp_eff) - max(mon_eff) + speed_tier + hp_ratio
    - Dynamically evaluates proactive switches when facing hard counters or stat drops
    - Sets entry hazards if opponent team >= 3
    - Clears entry hazards with Rapid Spin/Defog if teammates threatened
    - Uses setup moves at 100% HP when possessing advantage
    - Ranks attacks by base_power * STAB * stat_ratio * accuracy * type_effectiveness
    """

    ENTRY_HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
    ANTI_HAZARDS = {"rapidspin", "defog", "mortalspin", "tidyup"}

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon) -> float:
        opp_types = opponent.active_types
        mon_types = mon.active_types

        opp_max_mult = 1.0
        for ot in opp_types:
            if ot:
                eff = get_type_effectiveness(ot, mon_types[0], mon_types[1])
                if eff > opp_max_mult:
                    opp_max_mult = eff

        mon_max_mult = 1.0
        for mt in mon_types:
            if mt:
                eff = get_type_effectiveness(mt, opp_types[0], opp_types[1])
                if eff > mon_max_mult:
                    mon_max_mult = eff

        score = opp_max_mult - mon_max_mult

        mon_spe = mon.effective_stat("spe")
        opp_spe = opponent.effective_stat("spe")
        if mon_spe > opp_spe:
            score += 0.1
        elif opp_spe > mon_spe:
            score -= 0.1

        score += mon.hp_percent * 0.5
        score -= opponent.hp_percent * 0.5
        return score

    def _should_switch_out(self, side: BattleSide, opp_active: Pokemon) -> bool:
        active = side.active_pokemon
        if not active or not opp_active:
            return False

        switches = side.available_switches()
        if not switches:
            return False

        has_decent_switch = any(self._estimate_matchup(side.pokemon[i], opp_active) > 0 for i in switches)
        if not has_decent_switch:
            return False

        if active.boosts.get("def", 0) <= -3 or active.boosts.get("spd", 0) <= -3:
            return True
        if active.boosts.get("atk", 0) <= -3 and active.effective_stat("atk") >= active.effective_stat("spa"):
            return True
        if active.boosts.get("spa", 0) <= -3 and active.effective_stat("spa") >= active.effective_stat("atk"):
            return True
        if self._estimate_matchup(active, opp_active) < -1.0:
            return True
        return False

    def select_action(self, state: BattleState, player: int = 2) -> Action:
        side = state.p2 if player == 2 else state.p1
        opp_side = state.p1 if player == 2 else state.p2
        active = side.active_pokemon
        opp_active = opp_side.active_pokemon

        valid_actions = state.get_valid_actions(player=player)
        move_actions = [a for a in valid_actions if a.action_type == ActionType.MOVE]
        switch_actions = [a for a in valid_actions if a.action_type == ActionType.SWITCH]

        if not active or not opp_active:
            return valid_actions[0]

        # 1. Switch out on bad matchup
        if switch_actions and self._should_switch_out(side, opp_active):
            best_sw = switch_actions[0]
            best_score = -999.0
            for sw in switch_actions:
                tgt = sw.target_slot - 1
                if 0 <= tgt < len(side.pokemon):
                    sc = self._estimate_matchup(side.pokemon[tgt], opp_active)
                    if sc > best_score:
                        best_score = sc
                        best_sw = sw
            return best_sw

        if not move_actions:
            return switch_actions[0] if switch_actions else valid_actions[0]

        n_opp_alive = sum(1 for p in opp_side.pokemon if not p.is_fainted)
        n_our_alive = sum(1 for p in side.pokemon if not p.is_fainted)

        # 2. Entry hazards
        if n_opp_alive >= 3:
            for ma in move_actions:
                mid = getattr(ma, "move_id", "").lower().replace("-", "")
                if mid in self.ENTRY_HAZARDS and not (Hazard.STEALTH_ROCK in opp_side.hazards if mid == "stealthrock" else False):
                    return ma

        # 3. Hazard removal
        if len(side.hazards) > 0 and n_our_alive >= 2:
            for ma in move_actions:
                mid = getattr(ma, "move_id", "").lower().replace("-", "")
                if mid in self.ANTI_HAZARDS:
                    return ma

        # 4. Setup moves at 100% HP
        if active.hp_percent >= 0.99 and self._estimate_matchup(active, opp_active) > 0:
            for ma in move_actions:
                mid = getattr(ma, "move_id", "").lower().replace("-", "")
                if mid in ("swordsdance", "nastyplot", "dragondance", "calmmind"):
                    return ma

        # 5. Move selection: maximize base_power * STAB * stat_ratio * accuracy * type_multiplier
        physical_ratio = active.effective_stat("atk") / max(1, opp_active.effective_stat("def"))
        special_ratio = active.effective_stat("spa") / max(1, opp_active.effective_stat("spd"))

        best_move_act = move_actions[0]
        best_score = -1.0

        for ma in move_actions:
            slot = getattr(ma, "move_slot", 1) - 1
            if 0 <= slot < len(active.moves):
                mv = active.moves[slot]
                bp = mv.base_power
                if bp == 0:
                    bp = 40
                stab = 1.5 if mv.move_type in active.active_types else 1.0
                ratio = physical_ratio if mv.category == MoveCategory.PHYSICAL else special_ratio
                acc = mv.accuracy / 100.0 if mv.accuracy > 0 else 1.0
                eff = get_type_effectiveness(mv.move_type, opp_active.active_types[0], opp_active.active_types[1])
                score = bp * stab * ratio * acc * eff
                if score > best_score:
                    best_score = score
                    best_move_act = ma

        return best_move_act


class TacticalHeuristicPlayer:
    """1-Ply heuristic evaluation modeling poke-env SimpleHeuristicsPlayer."""

    def __init__(self):
        self.heuristic = HeuristicEvaluator()

    def select_action(self, state: BattleState, player: int = 2) -> Action:
        inv_state = BattleState(
            p1=state.p2 if player == 2 else state.p1,
            p2=state.p1 if player == 2 else state.p2,
            weather=state.weather,
            terrain=state.terrain,
            turn=state.turn
        )
        valid_actions = inv_state.get_valid_actions(player=1)
        best_act = valid_actions[0]
        best_val = -9999.0

        for a in valid_actions:
            sim = simulate_turn_transition(inv_state, a, None)
            v = self.heuristic.evaluate(sim)
            if v > best_val:
                best_val = v
                best_act = a
        return best_act


class ExpectiminimaxPlayer:
    """2-Ply Expectiminimax with heuristic scoring (Philip Mariglia's ShowdownAI architecture)."""

    def __init__(self, depth: int = 2):
        self.heuristic = HeuristicEvaluator()
        self.resolver = SubgameResolver(evaluator=self.heuristic)
        self.depth = depth

    def select_action(self, state: BattleState, player: int = 2) -> Action:
        inv_state = BattleState(
            p1=state.p2 if player == 2 else state.p1,
            p2=state.p1 if player == 2 else state.p2,
            weather=state.weather,
            terrain=state.terrain,
            turn=state.turn
        )
        act, _, _, _ = self.resolver.resolve_turn(inv_state, depth=self.depth, sample=False)
        return act


class GrandmasterBCPlayer:
    """Pure neural imitation policy trained on 1800+ human ladder replays."""

    def __init__(self, checkpoint_path: str = "checkpoints/glaubermon_max_elite.pt", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = GlaubermonMaxNet(d_model=256, nhead=8, num_actions=14).to(self.device)
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.evaluator = NeuralEvaluator(self.model, self.device)

    def select_action(self, state: BattleState, player: int = 2) -> Action:
        inv_state = BattleState(
            p1=state.p2 if player == 2 else state.p1,
            p2=state.p1 if player == 2 else state.p2,
            weather=state.weather,
            terrain=state.terrain,
            turn=state.turn
        )
        actions = inv_state.get_valid_actions(player=1)
        priors = self.evaluator.get_policy_prior(inv_state, actions)
        return actions[int(np.argmax(priors))]


class HistoricalAlphaZeroPlayer:
    """Prior AlphaZero checkpoint (1-ply subgame search with historical weights)."""

    def __init__(self, checkpoint_path: str = "checkpoints/glaubermon_alphazero_latest.pt", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = GlaubermonMaxNet(d_model=256, nhead=8, num_actions=14).to(self.device)
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.evaluator = NeuralEvaluator(self.model, self.device)
        self.resolver = SubgameResolver(evaluator=self.evaluator)

    def select_action(self, state: BattleState, player: int = 2) -> Action:
        inv_state = BattleState(
            p1=state.p2 if player == 2 else state.p1,
            p2=state.p1 if player == 2 else state.p2,
            weather=state.weather,
            terrain=state.terrain,
            turn=state.turn
        )
        act, _, _, _ = self.resolver.resolve_turn(inv_state, depth=1, sample=False)
        return act


class PredictiveExploiterPlayer:
    """Aggressive anti-meta player that predicts opponent switches and clicks coverage."""

    def __init__(self):
        self.heuristic = HeuristicEvaluator()

    def select_action(self, state: BattleState, player: int = 2) -> Action:
        inv_state = BattleState(
            p1=state.p2 if player == 2 else state.p1,
            p2=state.p1 if player == 2 else state.p2,
            weather=state.weather,
            terrain=state.terrain,
            turn=state.turn
        )
        p1_act = inv_state.p1.active_pokemon
        p2_act = inv_state.p2.active_pokemon
        valid_actions = inv_state.get_valid_actions(player=1)

        if p1_act and p2_act:
            opp_weak = False
            for ot in p2_act.active_types:
                for mt in p1_act.active_types:
                    if ot and mt and get_type_effectiveness(mt, ot) > 1.0:
                        opp_weak = True
                        break

            opp_switches = inv_state.p2.available_switches()
            if opp_weak and opp_switches:
                best_opp_sw = inv_state.p2.pokemon[opp_switches[0]]
                best_act = valid_actions[0]
                best_dmg = -1
                for a in valid_actions:
                    if a.action_type == ActionType.MOVE:
                        slot = a.move_slot - 1
                        if 0 <= slot < len(p1_act.moves):
                            mv = p1_act.moves[slot]
                            rolls = calculate_damage_rolls(p1_act, best_opp_sw, mv, inv_state.weather, inv_state.terrain)
                            dmg = rolls[len(rolls) // 2]
                            if dmg > best_dmg:
                                best_dmg = dmg
                                best_act = a
                if best_dmg > 0:
                    return best_act

        best_act = valid_actions[0]
        best_val = -9999.0
        for a in valid_actions:
            sim = simulate_turn_transition(inv_state, a, None)
            v = self.heuristic.evaluate(sim)
            if v > best_val:
                best_val = v
                best_act = a
        return best_act


def choose_smart_faint_switch(side: BattleSide, opp_active: Optional[Pokemon]) -> int:
    """Select the best bench Pokemon to switch into when active Pokemon faints."""
    switches = side.available_switches()
    if not switches:
        return 0
    if not opp_active:
        return switches[0]
    best_slot = switches[0]
    best_score = -999.0
    for s_idx in switches:
        mon = side.pokemon[s_idx]
        score = mon.hp_percent * 2.0
        t1, t2 = mon.active_types
        for ot in opp_active.active_types:
            if ot:
                score -= get_type_effectiveness(ot, t1, t2)
        if score > best_score:
            best_score = score
            best_slot = s_idx
    return best_slot


class BenchmarkTournament:
    """Runs automated tournament battles against baseline archetypes with full blunder auditing."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/glaubermon_rebel_latest.pt",
        depth: int = 1,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.depth = depth

        # Load main model with HybridEvaluator (Neural + Heuristic) matching production bot
        self.model = GlaubermonMaxNet(d_model=256, nhead=8, num_actions=14).to(self.device)
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"Loaded Glaubermon Max model: {checkpoint_path}")
        else:
            print(f"Checkpoint not found ({checkpoint_path}), using random initialization")
        self.neural_eval = NeuralEvaluator(self.model, self.device)
        self.heuristic_eval = HeuristicEvaluator()
        self.evaluator = HybridEvaluator(self.neural_eval, self.heuristic_eval, weight_neural=0.60)
        self.resolver = SubgameResolver(evaluator=self.evaluator)

        # Initialize opponent bot pool
        self.opponents = {
            "poke_env_heuristics": PokeEnvSimpleHeuristicsPlayer(),
            "poke_env_max_bp": PokeEnvMaxBasePowerPlayer(),
            "tactical": TacticalHeuristicPlayer(),
            "expectiminimax": ExpectiminimaxPlayer(depth=2),
            "grandmaster_bc": GrandmasterBCPlayer(device=device),
            "alphazero": HistoricalAlphaZeroPlayer(device=device),
            "predictive": PredictiveExploiterPlayer(),
        }

    def run_match(self, opponent_name: str = "tactical", mirror: bool = True) -> Dict:
        """Run a single match and return detailed match statistics and blunder metrics."""
        if mirror:
            state = BattleState(
                p1=BattleSide(pokemon=get_meta_team_balance()),
                p2=BattleSide(pokemon=get_meta_team_balance())
            )
        else:
            state = generate_competitive_battle()

        turns = 0
        search_times = []
        p1_switches = 0
        p1_tera_turn = None

        suicide_switches = 0
        purposeless_whirlwinds = 0
        lethal_setup_blunders = 0
        consecutive_protects = 0

        opp_player = self.opponents.get(opponent_name, self.opponents["tactical"])

        while not state.is_game_over and turns < 40:
            turns += 1

            p1_active_before = state.p1.active_pokemon
            p2_active_before = state.p2.active_pokemon

            # 1. Glaubermon Max Turn (P1)
            t0 = time.time()
            act1, strat1, acts1, val1 = self.resolver.resolve_turn(state, depth=self.depth, sample=False)
            t1 = time.time()
            search_times.append(t1 - t0)

            # Audit Blunder: Lethal Setup
            m1_id = getattr(act1, "move_id", "").lower().replace("-", "") if act1.action_type == ActionType.MOVE else ""
            if m1_id in ("swordsdance", "nastyplot", "calmmind", "dragondance", "quiverdance", "shellsmash"):
                if p1_active_before and p2_active_before:
                    for a in acts1:
                        if a.action_type == ActionType.MOVE:
                            mv = next((x for x in p1_active_before.moves if getattr(x, "id", "") == a.move_id), None)
                            if mv and mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                                rolls = calculate_damage_rolls(p1_active_before, p2_active_before, mv, state.weather, state.terrain)
                                if rolls and rolls[0] >= p2_active_before.current_hp:
                                    lethal_setup_blunders += 1
                                    break

            # Audit Blunder: Purposeless Whirlwind
            if m1_id in ("whirlwind", "roar") and p2_active_before:
                p2_has_boosts = any(v > 0 for v in p2_active_before.boosts.values())
                p2_has_hazards = len(state.p2.hazards) > 0
                if not p2_has_boosts and not p2_has_hazards:
                    purposeless_whirlwinds += 1

            # Audit Blunder: Consecutive Protect Loop
            if m1_id in ("protect", "spikyshield", "detect", "banefulbunker"):
                streak = getattr(p1_active_before, "protect_streak", 0) if p1_active_before else 0
                if streak >= 2:
                    consecutive_protects += 1

            if act1.action_type == ActionType.SWITCH:
                p1_switches += 1
            if getattr(act1, "is_tera", False) and p1_tera_turn is None:
                p1_tera_turn = turns

            # 2. Opponent Action (P2)
            act2 = opp_player.select_action(state, player=2)

            # 3. Simulate turn transition
            old_p1_hp = {i: p.current_hp for i, p in enumerate(state.p1.pokemon)}
            state = simulate_turn_transition(state, act1, act2)

            # Audit Blunder: Suicide Switch
            if act1.action_type == ActionType.SWITCH:
                tgt = getattr(act1, "target_slot", 1) - 1
                if 0 <= tgt < len(state.p1.pokemon):
                    sw_mon = state.p1.pokemon[tgt]
                    hp_lost = old_p1_hp.get(tgt, sw_mon.max_hp) - sw_mon.current_hp
                    if sw_mon.is_fainted or hp_lost >= sw_mon.max_hp * 0.80:
                        suicide_switches += 1

            # Smart switch on faints
            if state.p1.active_pokemon and state.p1.active_pokemon.is_fainted and not state.p1.is_all_fainted:
                best_sw1 = choose_smart_faint_switch(state.p1, state.p2.active_pokemon)
                state.p1.active_index = best_sw1
            if state.p2.active_pokemon and state.p2.active_pokemon.is_fainted and not state.p2.is_all_fainted:
                best_sw2 = choose_smart_faint_switch(state.p2, state.p1.active_pokemon)
                state.p2.active_index = best_sw2

        is_win = (state.winner == 1)
        return {
            "won": is_win,
            "turns": turns,
            "switches": p1_switches,
            "tera_turn": p1_tera_turn,
            "mean_search_time": float(np.mean(search_times)) if search_times else 0.0,
            "suicide_switches": suicide_switches,
            "purposeless_whirlwinds": purposeless_whirlwinds,
            "lethal_setup_blunders": lethal_setup_blunders,
            "consecutive_protects": consecutive_protects
        }

    def benchmark(
        self,
        opponent_types: List[str] = ["greedy", "tactical", "expectiminimax", "grandmaster_bc", "alphazero", "predictive"],
        games_per_type: int = 10,
        mirror: bool = True
    ) -> Dict:
        """Run full tournament across specified archetypes with automated blunder auditing."""
        results = {}

        matchup_mode = "Symmetric (Gen 9 OU Balance Mirror)" if mirror else "Random Meta Pool Drafts"
        print("=" * 80)
        print("          GLAUBERMON MAX MULTI-BOT TOURNAMENT BENCHMARK")
        print(f"  Hardware: {self.device} | Subgame Depth: {self.depth} | Mode: {matchup_mode}")
        print(f"  Games per Archetype: {games_per_type} | Total Matchups: {len(opponent_types)}")
        print("=" * 80)

        total_blunders = {"suicide_switches": 0, "purposeless_whirlwinds": 0, "lethal_setup_blunders": 0, "consecutive_protects": 0}

        for opp in opponent_types:
            if opp not in self.opponents:
                continue
            print(f"\n--- Round: GLAUBERMON MAX vs {opp.upper()} ({games_per_type} games) ---")
            wins = 0
            all_turns = []
            all_switches = []
            all_times = []

            for _ in tqdm(range(games_per_type), desc=f"Battles vs {opp}"):
                stats = self.run_match(opponent_name=opp, mirror=mirror)
                if stats["won"]:
                    wins += 1
                all_turns.append(stats["turns"])
                all_switches.append(stats["switches"])
                all_times.append(stats["mean_search_time"])
                for k in total_blunders:
                    total_blunders[k] += stats.get(k, 0)

            wr = wins / games_per_type
            results[opp] = {
                "winrate": wr,
                "wins": wins,
                "total": games_per_type,
                "avg_turns": float(np.mean(all_turns)),
                "avg_switches": float(np.mean(all_switches)),
                "avg_search_time_sec": float(np.mean(all_times))
            }
            print(f"  Result vs {opp.upper()}: {wins}/{games_per_type} Wins ({wr * 100:.1f}%) | Avg Turns: {np.mean(all_turns):.1f} | Avg Move Time: {np.mean(all_times):.3f}s")

        print("\n" + "=" * 80)
        print("                 TOURNAMENT SUMMARY TABLE")
        print("=" * 80)
        print(f"  {'Opponent Bot':<25} | {'Win Rate':<10} | {'Record':<8} | {'Avg Turns':<10} | {'Move Time':<10}")
        print("  " + "-" * 74)
        for opp, data in results.items():
            print(f"  {opp:<25} | {data['winrate'] * 100:>6.1f}%   | {data['wins']}/{data['total']:<6} | {data['avg_turns']:>8.1f}   | {data['avg_search_time_sec']:>7.3f}s")

        print("\n" + "=" * 80)
        print("                 AUTOMATED BLUNDER AUDIT REPORT")
        print("=" * 80)
        print(f"  Suicide Switches (lost >=80% HP on switch): {total_blunders['suicide_switches']}")
        print(f"  Purposeless Whirlwind/Roar (0 hazards, 0 boosts): {total_blunders['purposeless_whirlwinds']}")
        print(f"  Lethal Setup Blunders (setup when lethal KO exists): {total_blunders['lethal_setup_blunders']}")
        print(f"  Protect Loops (consecutive protect >= 2): {total_blunders['consecutive_protects']}")
        all_clear = sum(total_blunders.values()) == 0
        status_str = "CLEAN (Zero Blunders Detected!)" if all_clear else "REGRESSIONS FOUND"
        print(f"  Overall Engine Status: {status_str}")
        print("=" * 80 + "\n")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glaubermon Max Automated Multi-Bot Tournament Suite")
    parser.add_argument("--games", type=int, default=5, help="Number of games per opponent archetype")
    parser.add_argument("--depth", type=int, default=1, help="Lookahead search depth (1 or 2)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/glaubermon_rebel_latest.pt")
    parser.add_argument("--opponents", nargs="+", default=["poke_env_heuristics", "poke_env_max_bp", "expectiminimax", "predictive"])
    parser.add_argument("--mirror", action="store_true", default=True, help="Use symmetric Gen 9 OU Balance team for fair benchmark")
    args = parser.parse_args()

    tournament = BenchmarkTournament(checkpoint_path=args.checkpoint, depth=args.depth)
    tournament.benchmark(opponent_types=args.opponents, games_per_type=args.games, mirror=args.mirror)
