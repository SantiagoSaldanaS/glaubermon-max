"""Automated End-to-End Battle Sparring Simulator.

Runs full 6v6 tournament battles between Glaubermon Max and competitive
opponent archetypes, logging and auditing every turn for gameplay quality,
long-term strategy, and blunder elimination.
"""

import sys
import logging
from typing import List, Tuple, Dict, Optional
from glaubermon.core.types import PokemonType, MoveCategory, ActionType, StatusCondition, Hazard
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.actions import Action, MoveAction, SwitchAction
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.data.meta_teams import get_meta_team_balance, get_meta_pokemon_by_species, build_meta_pokemon
from glaubermon.search.subgame_resolver import SubgameResolver, simulate_turn_transition
from glaubermon.search.evaluators import HeuristicEvaluator, HybridEvaluator
from glaubermon.inference.damage_calc import calculate_damage_rolls
from glaubermon.core.constants import clean_key, get_type_effectiveness


class SparringAuditor:
    def __init__(self):
        self.blunders: List[str] = []
        self.warnings: List[str] = []
        self.turn_log: List[str] = []

    def audit_bot_turn(self, turn: int, state: BattleState, action: Action):
        p1_active = state.p1.active_pokemon
        p2_active = state.p2.active_pokemon
        p1_fallen = sum(1 for p in state.p1.pokemon if p.is_fainted)

        # 1. Check for Damaging Move into Known Type Immunity
        if action.action_type == ActionType.MOVE and p1_active and p2_active:
            mv = next((m for m in p1_active.moves if getattr(m, "id", "") == action.move_id), None)
            if mv and mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                rolls = calculate_damage_rolls(p1_active, p2_active, mv, state.weather, state.terrain, attacker_side=state.p1)
                if rolls and max(rolls) == 0:
                    blunder_msg = f"[Turn {turn} BLUNDER] Bot {p1_active.species} clicked {mv.name} dealing 0 damage into {p2_active.species} ({p2_active.active_types})!"
                    self.blunders.append(blunder_msg)

        # 2. Check for Early Kingambit Switch-in
        if action.action_type == ActionType.SWITCH:
            tgt_idx = getattr(action, "target_slot", 1) - 1
            if 0 <= tgt_idx < len(state.p1.pokemon):
                tgt = state.p1.pokemon[tgt_idx]
                if clean_key(tgt.species) == "kingambit" and p1_fallen < 2 and p1_active and not p1_active.is_fainted:
                    blunder_msg = f"[Turn {turn} BLUNDER] Bot voluntarily switched into Kingambit early with only {p1_fallen} fallen allies!"
                    self.blunders.append(blunder_msg)

        # 3. Check for Terastallizing on low-HP target
        if action.action_type == ActionType.MOVE and getattr(action, "is_tera", False) and p2_active:
            if p2_active.current_hp <= 40 or p2_active.hp_percent <= 0.20:
                blunder_msg = f"[Turn {turn} BLUNDER] Bot burned Terastallization on {p2_active.species} who has only {p2_active.current_hp} HP ({p2_active.hp_percent*100:.1f}%)!"
                self.blunders.append(blunder_msg)


def build_pelol94_team() -> List[Pokemon]:
    """Exact team fielded by Pelol94: Gliscor / Great Tusk / Kingambit / Ting-Lu / Iron Valiant / Dragapult."""
    return [
        get_meta_pokemon_by_species("Gliscor"),
        get_meta_pokemon_by_species("Great Tusk"),
        get_meta_pokemon_by_species("Kingambit"),
        get_meta_pokemon_by_species("Ting-Lu"),
        get_meta_pokemon_by_species("Iron Valiant"),
        get_meta_pokemon_by_species("Dragapult"),
    ]


def run_sparring_match(max_turns: int = 40) -> Tuple[bool, List[str], List[str]]:
    """Simulate a complete 6v6 match between Bot and the Pelol94 team."""
    bot_team = get_meta_team_balance()
    opp_team = build_pelol94_team()

    state = BattleState(
        p1=BattleSide(pokemon=bot_team, active_index=0),
        p2=BattleSide(pokemon=opp_team, active_index=4)  # Iron Valiant lead (Booster Speed)
    )
    # Activate Iron Valiant Booster Speed
    state.p2.active_pokemon.boosts["spe"] = 1

    bot_resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    opp_resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    auditor = SparringAuditor()

    print(f"\n=======================================================")
    print(f"  STARTING AUTONOMOUS SPARRING MATCH: Bot vs Pelol94 Team")
    print(f"  Bot Lead: {state.p1.active_pokemon.species}")
    print(f"  Opp Lead: {state.p2.active_pokemon.species}")
    print(f"=======================================================\n")

    for turn in range(1, max_turns + 1):
        if state.is_game_over:
            break

        # Handle faint replacements if active Pokémon are fainted
        if state.p1.active_pokemon.is_fainted:
            avail = state.p1.available_switches()
            if not avail:
                break
            # Pick healthiest non-Kingambit switch if early, else best switch
            best_sw = avail[0]
            p1_fallen = sum(1 for p in state.p1.pokemon if p.is_fainted)
            for sw in avail:
                p = state.p1.pokemon[sw]
                if clean_key(p.species) == "kingambit" and p1_fallen < 3 and len(avail) > 1:
                    continue
                if p.current_hp > state.p1.pokemon[best_sw].current_hp:
                    best_sw = sw
            state.p1.active_index = best_sw
            print(f"[Turn {turn} Faint Switch] Bot sends out {state.p1.active_pokemon.species} ({state.p1.active_pokemon.current_hp}/{state.p1.active_pokemon.max_hp} HP)")

        if state.p2.active_pokemon.is_fainted:
            avail = state.p2.available_switches()
            if not avail:
                break
            state.p2.active_index = avail[0]
            print(f"[Turn {turn} Faint Switch] Opponent sends out {state.p2.active_pokemon.species} ({state.p2.active_pokemon.current_hp}/{state.p2.active_pokemon.max_hp} HP)")

        # Resolve actions for both players
        a1, strat1, _, _ = bot_resolver.resolve_turn(state, depth=1, sample=False)
        a2, strat2, _, _ = opp_resolver.resolve_turn(state, depth=1, sample=False)

        if a1 is None or a2 is None:
            break

        # Audit Bot Action for blunders
        auditor.audit_bot_turn(turn, state, a1)

        # Log turn choices
        p1_mon = state.p1.active_pokemon.species
        p2_mon = state.p2.active_pokemon.species
        a1_desc = f"{a1.move_id}{' (Tera)' if getattr(a1, 'is_tera', False) else ''}" if a1.action_type == ActionType.MOVE else f"Switch -> {state.p1.pokemon[a1.target_slot-1].species}"
        a2_desc = f"{a2.move_id}{' (Tera)' if getattr(a2, 'is_tera', False) else ''}" if a2.action_type == ActionType.MOVE else f"Switch -> {state.p2.pokemon[a2.target_slot-1].species}"

        print(f"Turn {turn}: [{p1_mon}] {a1_desc}  vs  [{p2_mon}] {a2_desc}")

        # Transition state
        state = simulate_turn_transition(state, a1, a2)
        print(f"         Result -> Bot {state.p1.active_pokemon.species}: {state.p1.active_pokemon.current_hp}/{state.p1.active_pokemon.max_hp} HP | Opp {state.p2.active_pokemon.species}: {state.p2.active_pokemon.current_hp}/{state.p2.active_pokemon.max_hp} HP")

    winner = "Bot (Glaubermon Max)" if state.winner == 1 else ("Opponent (Pelol94 Team)" if state.winner == 2 else "Draw/Incomplete")
    print(f"\n=======================================================")
    print(f"  MATCH COMPLETED in {turn} turns! Winner: {winner}")
    print(f"  Bot Fainted: {state.p1.fainted_count}/6 | Opp Fainted: {state.p2.fainted_count}/6")
    print(f"  Blunders Detected: {len(auditor.blunders)}")
    for b in auditor.blunders:
        print(f"    - {b}")
    print(f"=======================================================\n")

    return len(auditor.blunders) == 0, auditor.blunders, auditor.turn_log


if __name__ == "__main__":
    success, blunders, logs = run_sparring_match()
    if not success:
        sys.exit(1)
