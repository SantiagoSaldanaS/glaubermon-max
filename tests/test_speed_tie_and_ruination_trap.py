"""Unit tests verifying speed tie risk asymmetry and Ruination stall trap elimination."""

import pytest
from glaubermon.data.meta_teams import META_TEAMS, get_meta_team_balance, get_meta_pokemon_by_species
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.search.subgame_resolver import SubgameResolver, apply_entry_hazards
from glaubermon.search.evaluators import HeuristicEvaluator
from glaubermon.core.actions import ActionType
from glaubermon.core.types import Hazard, StatusCondition
from glaubermon.scripts.mass_battle_auditor import execute_showdown_accurate_force_switch


def test_turn14_forced_switch_selects_kingambit_priority_kill():
    """Turn 14: Kingambit holds guaranteed priority Sucker Punch kill on 11 HP Dragapult.
    It must be selected over risking 100% Dragapult on a lethal speed tie."""
    p1_team = META_TEAMS["balance"]()
    p2_team = META_TEAMS["pelol94"]()
    p1_team[0].current_hp = 0  # Great Tusk fainted (fallen = 1)
    p1_team[1].current_hp = 315 # Gholdengo
    p1_team[2].current_hp = 341 # Kingambit
    p1_team[3].current_hp = 317 # Dragapult (100% HP)
    p1_team[4].current_hp = 12  # Ogerpon
    p1_team[5].current_hp = 200 # Ting-Lu
    p2_team[0].current_hp = 0  # Iron Valiant fainted
    p2_team[4].current_hp = 11  # Opponent Dragapult (3.5% HP, -2 SpA)
    p2_team[4].boosts["spa"] = -2

    state14 = BattleState(
        p1=BattleSide(pokemon=p1_team, active_index=0, hazards={Hazard.SPIKES_1: 3}),
        p2=BattleSide(pokemon=p2_team, active_index=4, hazards={})
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    chosen_slot = execute_showdown_accurate_force_switch(resolver, state14.p1, state14.p2.active_pokemon, state14)
    assert chosen_slot == 2, f"Expected Kingambit (slot 2) with priority lethal, got {chosen_slot} ({state14.p1.pokemon[chosen_slot].species})"


def test_ting_lu_ruination_stall_trap_avoids_ruination():
    """Ting-Lu poisoned vs Poison Heal Gliscor must NOT click Ruination; must pivot or phaze."""
    state5 = BattleState(
        p1=BattleSide(pokemon=META_TEAMS["balance"](), active_index=5),
        p2=BattleSide(pokemon=META_TEAMS["pelol94"](), active_index=1)
    )
    state5.p1.pokemon[5].status = StatusCondition.TOXIC
    state5.p2.pokemon[1].status = StatusCondition.TOXIC
    state5.p2.pokemon[1].current_hp = 150

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    act, strat, acts, val = resolver.resolve_turn(state5, depth=1, sample=False)
    ruin_prob = next((p for a, p in zip(acts, strat) if getattr(a, "move_id", "") == "ruination"), 0.0)
    assert ruin_prob < 0.001, f"Ruination into Poison Heal Gliscor while poisoned must be 0%, got {ruin_prob:.3f}"
    assert getattr(act, "move_id", "") != "ruination"


def test_speed_tie_asymmetry_penalizes_healthy_sweeper():
    """A healthy 100% HP Dragapult risking a 50/50 lethal speed tie against an 11 HP Dragapult
    must receive an asymmetry risk penalty compared to a risk-free counter."""
    s = BattleState(
        p1=BattleSide(pokemon=META_TEAMS["balance"](), active_index=3),
        p2=BattleSide(pokemon=META_TEAMS["pelol94"](), active_index=4)
    )
    s.p2.pokemon[4].current_hp = 11
    s.p2.pokemon[4].boosts["spa"] = -2

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    moves_only = [a for a in s.get_valid_actions(1) if a.action_type == ActionType.MOVE]
    _, _, _, val_draga = resolver.resolve_turn(s, depth=1, sample=False, p1_actions_override=moves_only)

    # Face-to-face value must reflect the dangerous negative-EV gamble
    assert val_draga < 0.0, f"Dragapult speed tie gamble must have negative value, got {val_draga:.3f}"
