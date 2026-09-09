"""Unit tests verifying engine fixes for endgame hallucinations, hazards, choice locks, and move sanity."""

import pytest
from glaubermon.core.types import PokemonType, MoveCategory, Hazard, ActionType
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.actions import MoveAction, SwitchAction
from glaubermon.data.showdown_dex import ShowdownDex
from glaubermon.search.subgame_resolver import SubgameResolver, simulate_turn_transition
from glaubermon.search.evaluators import HeuristicEvaluator


@pytest.fixture
def dex():
    return ShowdownDex.get_instance()


def test_stealth_rock_tera_flying_damage(dex):
    """Verify that a Tera Flying Kingambit takes 25% damage from Stealth Rock (not 6.25%)."""
    kingambit = Pokemon(
        species="Kingambit",
        types=(PokemonType.DARK, PokemonType.STEEL),
        max_hp=394,
        current_hp=62,
        tera_type=PokemonType.FLYING,
        is_terastallized=True
    )
    hazards = {Hazard.STEALTH_ROCK: 1}
    dmg = kingambit.calculate_hazard_damage(hazards)
    # 25% of 394 = 98 damage
    assert dmg == int(394 * 0.25)
    assert kingambit.is_dead_to_hazards(hazards) is True


def test_stealth_rock_base_kingambit_damage(dex):
    """Verify that non-terastallized Kingambit (Dark/Steel) takes 6.25% damage."""
    kingambit = Pokemon(
        species="Kingambit",
        types=(PokemonType.DARK, PokemonType.STEEL),
        max_hp=394,
        current_hp=62,
        is_terastallized=False
    )
    hazards = {Hazard.STEALTH_ROCK: 1}
    dmg = kingambit.calculate_hazard_damage(hazards)
    # 6.25% of 394 = 24 damage
    assert dmg == int(394 * 0.0625)
    assert kingambit.is_dead_to_hazards(hazards) is False


def test_choice_item_locking(dex):
    """Verify that using a move while holding Choice Specs locks the Pokémon into that move."""
    dragapult = Pokemon(
        species="Dragapult",
        types=(PokemonType.DRAGON, PokemonType.GHOST),
        max_hp=317,
        current_hp=317,
        item="choicespecs",
        moves=[dex.get_move("dracometeor"), dex.get_move("shadowball"), dex.get_move("flamethrower"), dex.get_move("uturn")]
    )
    tinglu = Pokemon(
        species="Ting-Lu",
        types=(PokemonType.DARK, PokemonType.GROUND),
        max_hp=514,
        current_hp=514,
        moves=[dex.get_move("earthquake")]
    )
    state = BattleState(
        p1=BattleSide(pokemon=[dragapult], active_index=0),
        p2=BattleSide(pokemon=[tinglu], active_index=0)
    )

    # Initially, all 4 moves are valid
    valid_acts = state.get_valid_actions(player=1)
    move_acts = [a for a in valid_acts if a.action_type == ActionType.MOVE]
    assert len(move_acts) == 4

    # Execute Flamethrower
    next_state = simulate_turn_transition(
        state,
        MoveAction(move_id="flamethrower", move_slot=3),
        MoveAction(move_id="earthquake", move_slot=1)
    )
    active_drag = next_state.p1.active_pokemon
    assert active_drag.choice_locked_move == "flamethrower"

    # Now, only Flamethrower is legal among moves
    locked_valid = next_state.get_valid_actions(player=1)
    locked_moves = [a for a in locked_valid if a.action_type == ActionType.MOVE]
    assert len(locked_moves) == 1
    assert locked_moves[0].move_id == "flamethrower"


def test_turn17_gholdengo_shadow_ball(dex):
    """Verify that Gholdengo uses 2x super-effective Shadow Ball over 0.5x resisted Make It Rain."""
    p1_gholdengo = Pokemon(
        species="Gholdengo",
        types=(PokemonType.STEEL, PokemonType.GHOST),
        current_hp=296,
        max_hp=315,
        moves=[dex.get_move("makeitrain"), dex.get_move("shadowball"), dex.get_move("nastyplot"), dex.get_move("recover")],
        raw_stats={"hp": 315, "atk": 140, "def": 226, "spa": 399, "spd": 218, "spe": 267}
    )
    p2_gholdengo = Pokemon(
        species="Gholdengo",
        types=(PokemonType.STEEL, PokemonType.GHOST),
        current_hp=103,
        max_hp=315,
        moves=[dex.get_move("makeitrain"), dex.get_move("shadowball"), dex.get_move("recover")],
        raw_stats={"hp": 315, "atk": 140, "def": 226, "spa": 399, "spd": 218, "spe": 267}
    )
    p2_kingambit = Pokemon(
        species="Kingambit",
        types=(PokemonType.DARK, PokemonType.STEEL),
        current_hp=341,
        max_hp=341,
        moves=[dex.get_move("kowtowcleave")]
    )

    state = BattleState(
        p1=BattleSide(pokemon=[p1_gholdengo], active_index=0, is_tera_used=True),
        p2=BattleSide(pokemon=[p2_gholdengo, p2_kingambit], active_index=0, is_tera_used=False)
    )

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action_d1, strat_d1, acts_d1, _ = resolver.resolve_turn(state, depth=1)
    assert getattr(action_d1, "move_id", "") == "shadowball"

    action_d2, strat_d2, acts_d2, _ = resolver.resolve_turn(state, depth=2)
    assert getattr(action_d2, "move_id", "") == "shadowball"


def test_turn15_kingambit_sucker_punch(dex):
    """Verify that Kingambit at 62 HP with Stealth Rock on field clicks Sucker Punch instead of switching out."""
    p1_kingambit = Pokemon(
        species="Kingambit",
        types=(PokemonType.DARK, PokemonType.STEEL),
        current_hp=62,
        max_hp=394,
        moves=[dex.get_move("kowtowcleave"), dex.get_move("suckerpunch"), dex.get_move("ironhead"), dex.get_move("swordsdance")],
        tera_type=PokemonType.FLYING,
        is_terastallized=True
    )
    p1_gholdengo = Pokemon(
        species="Gholdengo",
        types=(PokemonType.STEEL, PokemonType.GHOST),
        current_hp=315,
        max_hp=315,
        moves=[dex.get_move("makeitrain"), dex.get_move("shadowball")]
    )
    p2_gholdengo = Pokemon(
        species="Gholdengo",
        types=(PokemonType.STEEL, PokemonType.GHOST),
        current_hp=315,
        max_hp=315,
        item="airballoon",
        moves=[dex.get_move("shadowball"), dex.get_move("makeitrain")]
    )

    state = BattleState(
        p1=BattleSide(pokemon=[p1_kingambit, p1_gholdengo], active_index=0, is_tera_used=True, hazards={Hazard.STEALTH_ROCK: 1}),
        p2=BattleSide(pokemon=[p2_gholdengo], active_index=0, is_tera_used=False)
    )

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, acts, _ = resolver.resolve_turn(state, depth=2)
    # At depth 2, Kingambit must not switch to preserve a dead mon; it should click Sucker Punch
    assert getattr(action, "move_id", "") == "suckerpunch"
