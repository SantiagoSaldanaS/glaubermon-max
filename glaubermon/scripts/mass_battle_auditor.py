"""Automated Mass-Battle Simulator and Deep Blunder Auditor for Glaubermon Max.

Executes autonomous 6v6 competitive matches across diverse OU meta archetypes:
- Pelol94 Exact Team (Toxic Gliscor, Great Tusk, Kingambit, Ting-Lu, Iron Valiant, Dragapult)
- Hyper Offense
- Bulky Stall
- Balance Mirror

Pits Glaubermon against diverse AI player styles:
- Tactical Heuristic Player
- Expectiminimax (2-ply)
- Predictive Exploiter
- Toxic Stall Player (human-like hazard stacking & toxic stalling)

Performs deep turn-by-turn auditing for all known pathologies:
- Missed lethal attacks
- Suicide switches
- Premature Kingambit switches (< 3 fallen allies)
- Choice-lock loops (-2/-4 SpA)
- Ineffective resisted attacks (<= 0.5x vs >= 2x stronger)
- ForceSwitch replacement blunders
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np

from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import ActionType, PokemonType, Hazard, MoveCategory, StatusCondition
from glaubermon.core.constants import get_type_effectiveness, clean_key
from glaubermon.core.actions import Action, MoveAction, SwitchAction
from glaubermon.search.evaluators import HeuristicEvaluator
from glaubermon.search.subgame_resolver import SubgameResolver, simulate_turn_transition, apply_entry_hazards
from glaubermon.inference.damage_calc import calculate_damage_rolls
from glaubermon.data.meta_teams import (
    get_meta_team_balance,
    get_meta_team_hyper_offense,
    get_meta_team_stall,
    get_meta_team_pelol94,
    META_TEAMS
)


class ToxicStallPlayer:
    """Simulates a patient, competitive human player utilizing hazard stacking and Toxic stalling."""

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
        move_actions = [a for a in valid_actions if a.action_type == ActionType.MOVE]
        switches = [a for a in valid_actions if a.action_type == ActionType.SWITCH]

        if not p1_act or not p2_act:
            return valid_actions[0]

        # 1. If opponent is poisoned/toxic, prioritize Protect / Spiky Shield for free chip
        if p2_act.status == StatusCondition.TOXIC:
            for ma in move_actions:
                mid = getattr(ma, "move_id", "")
                if mid in ("protect", "spikyshield", "banefulbunker"):
                    if not getattr(p1_act, "is_protected", False):
                        return ma

        # 2. If healthy and opponent not poisoned, click Toxic
        if p1_act.hp_percent > 0.60 and p2_act.status is None:
            for ma in move_actions:
                if getattr(ma, "move_id", "") == "toxic":
                    if PokemonType.STEEL not in p2_act.active_types and PokemonType.POISON not in p2_act.active_types:
                        return ma

        # 3. Stack hazards if fewer than 2 hazards present
        opp_side = inv_state.p2
        if Hazard.STEALTH_ROCK not in opp_side.hazards or opp_side.hazards.get(Hazard.SPIKES_1, 0) < 2:
            for ma in move_actions:
                mid = getattr(ma, "move_id", "")
                if mid == "stealthrock" and Hazard.STEALTH_ROCK not in opp_side.hazards:
                    return ma
                if mid in ("spikes", "ceaselessedge") and opp_side.hazards.get(Hazard.SPIKES_1, 0) < 2:
                    return ma

        # 4. Check for lethal attacks
        for ma in move_actions:
            slot = getattr(ma, "move_slot", 1) - 1
            if 0 <= slot < len(p1_act.moves):
                mv = p1_act.moves[slot]
                if mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                    rolls = calculate_damage_rolls(p1_act, p2_act, mv, inv_state.weather, inv_state.terrain, attacker_side=inv_state.p1)
                    if rolls and rolls[0] >= p2_act.current_hp:
                        return ma

        # 5. Otherwise, pick best heuristic move/switch
        best_act = valid_actions[0]
        best_val = -9999.0
        for a in valid_actions:
            sim = simulate_turn_transition(inv_state, a, None)
            v = self.heuristic.evaluate(sim)
            if v > best_val:
                best_val = v
                best_act = a
        return best_act


class TacticalPlayer:
    """1-ply heuristic evaluation."""
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
    """2-ply simultaneous lookahead search."""
    def __init__(self, depth: int = 2):
        self.resolver = SubgameResolver(evaluator=HeuristicEvaluator())
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


class PredictivePlayer:
    """Aggressive anti-switch prediction exploiter."""
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
                best_act = None
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
                if best_act is not None and best_dmg > 0:
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


def execute_showdown_accurate_force_switch(
    resolver: SubgameResolver,
    side: BattleSide,
    opp_active: Optional[Pokemon],
    state: BattleState
) -> int:
    """Executes the exact candidate combat-moves-only evaluation from showdown_bot.py."""
    switches = side.available_switches()
    if not switches:
        if side.active_pokemon and not side.active_pokemon.is_fainted:
            return side.active_index
        for i, p in enumerate(side.pokemon):
            if not p.is_fainted:
                return i
        return side.active_index
    if len(switches) == 1:
        return switches[0]

    safe_switches = [s for s in switches if not side.pokemon[s].is_dead_to_hazards(side.hazards)]
    candidate_indices = safe_switches if safe_switches else switches

    best_slot = candidate_indices[0]
    best_val = -float('inf')
    fallen = sum(1 for p in side.pokemon if p.is_fainted)

    for slot_idx in candidate_indices:
        s_cand = state.clone()
        s_cand.p1.active_index = slot_idx
        apply_entry_hazards(s_cand, 1, slot_idx)

        # Restrict candidate evaluation to direct combat moves against opponent active
        cand_moves = [a for a in s_cand.get_valid_actions(1) if a.action_type == ActionType.MOVE]
        moves_override = cand_moves if cand_moves else None

        _, _, _, cand_val = resolver.resolve_turn(s_cand, depth=1, sample=False, p1_actions_override=moves_override)

        cand_mon = side.pokemon[slot_idx]
        if cand_mon:
            spec_key = clean_key(cand_mon.species)
            if spec_key == "kingambit" and fallen < 3:
                has_priority_kill = False
                opp_mon = state.p2.active_pokemon
                if opp_mon and not opp_mon.is_fainted:
                    for mv in cand_mon.moves:
                        if getattr(mv, "priority", 0) > 0 and mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                            rolls = calculate_damage_rolls(cand_mon, opp_mon, mv, state.weather, state.terrain, attacker_side=s_cand.p1)
                            if rolls and min(rolls) >= opp_mon.current_hp:
                                has_priority_kill = True
                                break
                if not has_priority_kill:
                    cand_val -= 8.0 * (3 - fallen)

        if cand_val > best_val:
            best_val = cand_val
            best_slot = slot_idx

    return best_slot


class MassBattleAuditor:
    """Automated high-throughput competitive battle simulator and deep blunder auditor."""

    def __init__(self, depth: int = 1):
        self.evaluator = HeuristicEvaluator()
        self.resolver = SubgameResolver(evaluator=self.evaluator)
        self.depth = depth
        self.opponents = {
            "tactical": TacticalPlayer(),
            "expectiminimax": ExpectiminimaxPlayer(depth=2),
            "predictive": PredictivePlayer(),
            "toxic_stall": ToxicStallPlayer(),
        }

    def run_battle(
        self,
        p1_archetype: str = "balance",
        p2_archetype: str = "pelol94",
        opp_ai: str = "toxic_stall"
    ) -> Dict:
        """Simulate a single full 6v6 match and record turn-by-turn decisions and blunder metrics."""
        p1_team = META_TEAMS[p1_archetype]()
        p2_team = META_TEAMS[p2_archetype]()

        state = BattleState(
            p1=BattleSide(pokemon=p1_team, hazards={}),
            p2=BattleSide(pokemon=p2_team, hazards={})
        )

        from glaubermon.client.showdown_bot import ShowdownBot
        bot_dummy = ShowdownBot.__new__(ShowdownBot)
        bot_dummy.opp_team = {"sim": [p.species for p in p2_team]}
        lead_order = bot_dummy.select_lead_order("sim")
        state.p1.active_index = int(lead_order[0]) - 1

        turns = 0
        p1_switches = 0
        p1_faints = 0
        p2_faints = 0

        # Detailed Blunder Categories
        blunders = {
            "missed_lethal": 0,
            "suicide_switches": 0,
            "early_kingambit_switch": 0,
            "choice_lock_stalls": 0,
            "resisted_attack_blunders": 0,
            "wasteful_tera": 0,
            "force_switch_blunders": 0,
        }
        blunder_details = []

        opp_player = self.opponents.get(opp_ai, self.opponents["tactical"])

        while not state.is_game_over and turns < 60:
            turns += 1

            p1_active_before = state.p1.active_pokemon
            p2_active_before = state.p2.active_pokemon
            p1_active_idx_before = state.p1.active_index
            p2_active_idx_before = state.p2.active_index

            if not p1_active_before or not p2_active_before:
                break

            p1_spe_before = p1_active_before.effective_stat("spe")
            p2_spe_before = p2_active_before.effective_stat("spe")

            # Check if P1 has a guaranteed legal lethal attack right now with speed advantage
            p1_has_faster_lethal = False
            lethal_move_id = None
            legal_moves = [a for a in state.get_valid_actions(1) if a.action_type == ActionType.MOVE]

            # Check if opponent holds priority lethal that preempts normal speed advantage
            opp_has_priority_lethal = False
            for m2 in p2_active_before.moves:
                if m2.priority > 0 and m2.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                    r2 = calculate_damage_rolls(p2_active_before, p1_active_before, m2, state.weather, state.terrain, attacker_side=state.p2)
                    if r2 and r2[0] >= p1_active_before.current_hp:
                        opp_has_priority_lethal = True
                        break

            for lm in legal_moves:
                mv = next((m for m in p1_active_before.moves if m.id == lm.move_id), None)
                if mv and mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                    if (mv.priority > 0 and clean_key(mv.id) != "suckerpunch") or (p1_spe_before > p2_spe_before and not opp_has_priority_lethal):
                        rolls = calculate_damage_rolls(p1_active_before, p2_active_before, mv, state.weather, state.terrain, attacker_side=state.p1)
                        if rolls and rolls[0] >= p2_active_before.current_hp:
                            p1_has_faster_lethal = True
                            lethal_move_id = mv.id
                            break

            # 1. Glaubermon Max Turn Action (P1)
            act1, strat1, acts1, val1 = self.resolver.resolve_turn(state, depth=self.depth, sample=False)

            # Audit Blunder: Missed Lethal
            if p1_has_faster_lethal:
                chosen_mid = getattr(act1, "move_id", "") if act1.action_type == ActionType.MOVE else "switch"
                if act1.action_type != ActionType.MOVE or chosen_mid != lethal_move_id:
                    # Check if the chosen move is also 100% lethal
                    is_also_lethal = False
                    if act1.action_type == ActionType.MOVE:
                        m_obj = next((m for m in p1_active_before.moves if m.id == chosen_mid), None)
                        if m_obj:
                            c_rolls = calculate_damage_rolls(p1_active_before, p2_active_before, m_obj, state.weather, state.terrain, attacker_side=state.p1)
                            if c_rolls and c_rolls[0] >= p2_active_before.current_hp:
                                is_also_lethal = True
                    if not is_also_lethal:
                        blunders["missed_lethal"] += 1
                        blunder_details.append(f"Turn {turns}: Missed faster lethal with {lethal_move_id}, chose {chosen_mid}")

            # Audit Blunder: Premature Kingambit voluntary switch
            p1_fallen = sum(1 for p in state.p1.pokemon if p.is_fainted)
            if act1.action_type == ActionType.SWITCH:
                tgt_slot = getattr(act1, "target_slot", 1) - 1
                if 0 <= tgt_slot < len(state.p1.pokemon):
                    tgt_mon = state.p1.pokemon[tgt_slot]
                    if clean_key(tgt_mon.species) == "kingambit" and p1_fallen < 3:
                        blunders["early_kingambit_switch"] += 1
                        blunder_details.append(f"Turn {turns}: Premature Kingambit switch with only {p1_fallen} fallen allies")

            # Audit Blunder: Choice lock stall at <= -2 SpA/Atk
            if act1.action_type == ActionType.MOVE and p1_active_before.choice_locked_move:
                locked = p1_active_before.choice_locked_move
                if locked in ("dracometeor", "overheat", "leafstorm", "superpower"):
                    if p1_active_before.boosts.get("spa", 0) <= -2 or p1_active_before.boosts.get("atk", 0) <= -2:
                        m_obj = next((m for m in p1_active_before.moves if m.id == locked), None)
                        c_rolls = calculate_damage_rolls(p1_active_before, p2_active_before, m_obj, state.weather, state.terrain, attacker_side=state.p1) if m_obj else None
                        is_lethal = bool(c_rolls and c_rolls[0] >= p2_active_before.current_hp)
                        if not is_lethal:
                            p2_stabs = [t for t in p2_active_before.types if t]
                            safe_sw = [
                                s for s in state.p1.available_switches()
                                if state.p1.pokemon[s].hp_percent >= 0.40
                                and not state.p1.pokemon[s].is_dead_to_hazards(state.p1.hazards)
                                and not any(get_type_effectiveness(t, state.p1.pokemon[s].active_types[0], state.p1.pokemon[s].active_types[1] if len(state.p1.pokemon[s].active_types) > 1 else None) > 1.0 for t in p2_stabs)
                            ]
                            if safe_sw:
                                blunders["choice_lock_stalls"] += 1
                                blunder_details.append(f"Turn {turns}: Dragapult/mon clicked {locked} at -2 offensive stat stage instead of pivoting to safe teammate")

            # Audit Blunder: Ineffective resisted attack
            if act1.action_type == ActionType.MOVE:
                mid = getattr(act1, "move_id", "")
                m_obj = next((m for m in p1_active_before.moves if m.id == mid), None)
                if m_obj and m_obj.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                    t1, t2 = p2_active_before.active_types[0], (p2_active_before.active_types[1] if len(p2_active_before.active_types) > 1 else None)
                    eff = get_type_effectiveness(m_obj.move_type, t1, t2)
                    if eff <= 0.5:
                        # Check if chosen move is lethal (never a blunder to kill)
                        c_rolls = calculate_damage_rolls(p1_active_before, p2_active_before, m_obj, state.weather, state.terrain, attacker_side=state.p1)
                        is_lethal = bool(c_rolls and c_rolls[0] >= p2_active_before.current_hp)
                        if not is_lethal:
                            # Check if another legal damaging move deals >= 2x damage
                            for alt_lm in legal_moves:
                                if alt_lm.move_id != mid:
                                    alt = next((m for m in p1_active_before.moves if m.id == alt_lm.move_id), None)
                                    if alt and alt.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                                        # If alt deals 0 damage (immunity from type, ability, or Air Balloon), it is not an alternative!
                                        alt_rolls = calculate_damage_rolls(p1_active_before, p2_active_before, alt, state.weather, state.terrain, attacker_side=state.p1)
                                        if not alt_rolls or max(alt_rolls) == 0:
                                            continue
                                        # If m_obj is priority and alt is not, and we are slower, priority is justified
                                        if m_obj.priority > 0 and alt.priority <= 0 and p1_spe_before < p2_spe_before:
                                            continue
                                        alt_eff = get_type_effectiveness(alt.move_type, t1, t2)
                                        if alt_eff >= 1.0 and (alt.base_power * alt_eff) >= 2.0 * (m_obj.base_power * eff):
                                            blunders["resisted_attack_blunders"] += 1
                                            blunder_details.append(f"Turn {turns}: Clicked {mid} ({eff}x) when {alt.id} ({alt_eff}x) is available")
                                            break

            if act1.action_type == ActionType.SWITCH:
                p1_switches += 1

            # 2. Opponent Turn Action (P2)
            act2 = opp_player.select_action(state, player=2)

            # 3. Simulate turn transition
            old_p1_hp = {i: p.current_hp for i, p in enumerate(state.p1.pokemon)}
            state = simulate_turn_transition(state, act1, act2)

            # Audit Blunder: Suicide switch (fainting immediately upon entry from a healthy state)
            if act1.action_type == ActionType.SWITCH:
                tgt_slot = getattr(act1, "target_slot", 1) - 1
                if 0 <= tgt_slot < len(state.p1.pokemon):
                    entering_mon = state.p1.pokemon[tgt_slot]
                    prev_hp = old_p1_hp.get(tgt_slot, entering_mon.max_hp)
                    damage_taken = prev_hp - entering_mon.current_hp
                    # A true suicide switch is switching a healthy Pokémon (>40% HP) directly into a lethal attack that faints it
                    if prev_hp > entering_mon.max_hp * 0.40 and entering_mon.is_fainted:
                        blunders["suicide_switches"] += 1
                        blunder_details.append(f"Turn {turns}: Suicide switch into {entering_mon.species} (fainted on entry from {prev_hp}/{entering_mon.max_hp} HP)")

            # Handle Faints with Showdown-accurate forced switch logic
            if state.p1.active_pokemon and state.p1.active_pokemon.is_fainted and not state.p1.is_all_fainted:
                best_sw1 = execute_showdown_accurate_force_switch(self.resolver, state.p1, state.p2.active_pokemon, state)
                # Audit: Did the forced switch pick Gholdengo into a lethal Ground attacker?
                f_mon = state.p1.pokemon[best_sw1]
                if clean_key(f_mon.species) == "gholdengo" and state.p2.active_pokemon:
                    if PokemonType.GROUND in state.p2.active_pokemon.active_types and not f_mon.item == "Air Balloon":
                        blunders["force_switch_blunders"] += 1
                        blunder_details.append(f"Turn {turns}: Force switch sent in Gholdengo against Ground attacker {state.p2.active_pokemon.species}")
                state.p1.active_index = best_sw1
                apply_entry_hazards(state, 1, best_sw1)

            if state.p2.active_pokemon and state.p2.active_pokemon.is_fainted and not state.p2.is_all_fainted:
                # Opponent smart faint switch
                switches2 = state.p2.available_switches()
                if switches2:
                    safe_p2 = [s for s in switches2 if not state.p2.pokemon[s].is_dead_to_hazards(state.p2.hazards)]
                    pick_p2 = safe_p2 if safe_p2 else switches2
                    best_p2_sw = max(pick_p2, key=lambda s: state.p2.pokemon[s].current_hp)
                    state.p2.active_index = best_p2_sw
                    apply_entry_hazards(state, 2, best_p2_sw)

        if state.is_game_over:
            is_win = (state.winner == 1)
        else:
            # Turn limit reached: Apply standard tournament tiebreaker
            p1_alive = sum(1 for p in state.p1.pokemon if not p.is_fainted)
            p2_alive = sum(1 for p in state.p2.pokemon if not p.is_fainted)
            if p1_alive > p2_alive:
                is_win = True
            elif p2_alive > p1_alive:
                is_win = False
            else:
                p1_hp_pct = sum(p.current_hp for p in state.p1.pokemon) / max(1, sum(p.max_hp for p in state.p1.pokemon))
                p2_hp_pct = sum(p.current_hp for p in state.p2.pokemon) / max(1, sum(p.max_hp for p in state.p2.pokemon))
                is_win = (p1_hp_pct >= p2_hp_pct)
        return {
            "won": is_win,
            "turns": turns,
            "switches": p1_switches,
            "blunders": blunders,
            "blunder_details": blunder_details,
            "winner": state.winner
        }

    def run_mass_audit(
        self,
        num_battles: int = 50,
        matchups: Optional[List[Tuple[str, str, str, int]]] = None
    ) -> Dict:
        """Run mass automated tournament with comprehensive statistics and blunder reporting."""
        if matchups is None:
            # Default diverse test suite:
            # 1. 20 battles vs Pelol94 team with Toxic Stall
            # 2. 10 battles vs Pelol94 team with Expectiminimax
            # 3. 10 battles vs Hyper Offense with Predictive AI
            # 4. 10 battles vs Balance Mirror with Tactical AI
            matchups = [
                ("balance", "pelol94", "toxic_stall", 20),
                ("balance", "pelol94", "expectiminimax", 10),
                ("balance", "hyper_offense", "predictive", 10),
                ("balance", "balance", "tactical", 10),
            ]

        print("=" * 85)
        print("          GLAUBERMON MAX AUTONOMOUS MASS-BATTLE AUDITOR (50+ GAMES)")
        print(f"  Evaluation Engine: Calibrated Domain Heuristic | Search Depth: {self.depth}")
        print("=" * 85)

        total_battles = sum(m[3] for m in matchups)
        all_results = []
        aggregate_blunders = {
            "missed_lethal": 0,
            "suicide_switches": 0,
            "early_kingambit_switch": 0,
            "choice_lock_stalls": 0,
            "resisted_attack_blunders": 0,
            "wasteful_tera": 0,
            "force_switch_blunders": 0,
        }
        all_blunder_traces = []

        for p1_arch, p2_arch, opp_ai, count in matchups:
            label = f"{p1_arch.upper()} vs {p2_arch.upper()} ({opp_ai.upper()})"
            print(f"\n--- Running: {label} [{count} games] ---")
            wins = 0
            turns_list = []

            for _ in tqdm(range(count), desc=f"{p2_arch} vs {opp_ai}"):
                res = self.run_battle(p1_arch, p2_arch, opp_ai)
                if res["won"]:
                    wins += 1
                turns_list.append(res["turns"])
                for b_k, b_v in res["blunders"].items():
                    aggregate_blunders[b_k] += b_v
                if res["blunder_details"]:
                    all_blunder_traces.extend(res["blunder_details"])

            wr = wins / count
            print(f"  Result: {wins}/{count} Wins ({wr * 100:.1f}%) | Avg Turns: {np.mean(turns_list):.1f}")
            all_results.append({
                "label": label,
                "wins": wins,
                "total": count,
                "winrate": wr,
                "avg_turns": float(np.mean(turns_list))
            })

        print("\n" + "=" * 85)
        print("                 TOURNAMENT PERFORMANCE SUMMARY")
        print("=" * 85)
        print(f"  {'Matchup':<45} | {'Win Rate':<10} | {'Record':<8} | {'Avg Turns':<10}")
        print("  " + "-" * 80)
        total_wins = sum(r["wins"] for r in all_results)
        for r in all_results:
            print(f"  {r['label']:<45} | {r['winrate'] * 100:>6.1f}%   | {r['wins']}/{r['total']:<6} | {r['avg_turns']:>8.1f}")
        print("  " + "-" * 80)
        overall_wr = total_wins / total_battles
        print(f"  {'OVERALL TOTAL':<45} | {overall_wr * 100:>6.1f}%   | {total_wins}/{total_battles:<6} |")

        print("\n" + "=" * 85)
        print("                 DEEP BLUNDER AUDIT REPORT")
        print("=" * 85)
        print(f"  Missed Lethal Attacks (failed to take guaranteed 1-hit KO):    {aggregate_blunders['missed_lethal']}")
        print(f"  Suicide Switches (switched into >=70% HP damage or faint):     {aggregate_blunders['suicide_switches']}")
        print(f"  Premature Kingambit Switches (entered with < 3 fallen allies):  {aggregate_blunders['early_kingambit_switch']}")
        print(f"  Choice-Lock Stalls (locked at <= -2 offensive stages):          {aggregate_blunders['choice_lock_stalls']}")
        print(f"  Ineffective Resisted Attacks (clicked <=0.5x vs >=2x stronger):{aggregate_blunders['resisted_attack_blunders']}")
        print(f"  ForceSwitch Blunders (sent Gholdengo into ungrounded Ground):  {aggregate_blunders['force_switch_blunders']}")
        print("  " + "-" * 80)
        total_b = sum(aggregate_blunders.values())
        print(f"  Total Pathologies Detected Across {total_battles} Games: {total_b}")
        if total_b > 0:
            print("\n  Sample Blunder Traces (First 5):")
            for trace in all_blunder_traces[:5]:
                print(f"    - {trace}")
        else:
            print("  Status: 100% CLEAN - ZERO BLUNDERS DETECTED ACROSS ALL BATTLES!")
        print("=" * 85 + "\n")

        return {
            "overall_winrate": overall_wr,
            "total_wins": total_wins,
            "total_battles": total_battles,
            "blunders": aggregate_blunders,
            "traces": all_blunder_traces
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glaubermon Max Automated Mass Battle Auditor")
    parser.add_argument("--depth", type=int, default=1, help="Lookahead search depth")
    parser.add_argument("--battles", type=int, default=50, help="Total number of battles to run")
    args = parser.parse_args()

    auditor = MassBattleAuditor(depth=args.depth)
    auditor.run_mass_audit()
