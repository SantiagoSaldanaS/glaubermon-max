"""Unit tests verifying tactical endgame fixes:
1. Mate-in-1 priority (Great Tusk Close Combat vs Kingambit)
2. Protect streak discipline (Ogerpon Ivy Cudgel vs Spiky Shield loop)
3. Defensive counter-pivot (Gholdengo -1 SpA vs Ground Earthquake)
4. Defensive Terastallization on setup (Kingambit Tera Flying vs Ground)
"""

import pytest
from glaubermon.data.showdown_dex import ShowdownDex
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.types import PokemonType, ActionType
from glaubermon.search.subgame_resolver import SubgameResolver
from glaubermon.search.evaluators import HeuristicEvaluator, HybridEvaluator


@pytest.fixture
def dex():
    return ShowdownDex.get_instance()


def test_great_tusk_mate_in_1_close_combat(dex):
    """Great Tusk with Booster Energy must select 4x super-effective Close Combat over resisted Ice Spinner/Rapid Spin."""
    tusk = Pokemon(
        species="Great Tusk",
        types=(PokemonType.GROUND, PokemonType.FIGHTING),
        current_hp=371,
        max_hp=371,
        raw_stats={"hp": 371, "atk": 397, "def": 298, "spa": 127, "spd": 142, "spe": 273},
        moves=[dex.get_move("closecombat"), dex.get_move("headlongrush"), dex.get_move("icespinner"), dex.get_move("rapidspin")],
        tera_type=PokemonType.ICE,
        is_terastallized=True
    )
    p1_ogerpon = Pokemon(species="Ogerpon-Wellspring", types=(PokemonType.GRASS, PokemonType.WATER), current_hp=301, max_hp=301, moves=[dex.get_move("ivycudgel")])
    p1_dragapult = Pokemon(species="Dragapult", types=(PokemonType.DRAGON, PokemonType.GHOST), current_hp=21, max_hp=317, moves=[dex.get_move("shadowball")])

    kingambit = Pokemon(
        species="Kingambit",
        types=(PokemonType.DARK, PokemonType.STEEL),
        current_hp=341,
        max_hp=341,
        raw_stats={"hp": 341, "atk": 370, "def": 276, "spa": 140, "spd": 206, "spe": 136},
        moves=[dex.get_move("kowtowcleave"), dex.get_move("suckerpunch"), dex.get_move("ironhead"), dex.get_move("swordsdance")],
    )

    p1_side = BattleSide(pokemon=[tusk, p1_ogerpon, p1_dragapult], active_index=0, is_tera_used=True)
    p2_side = BattleSide(pokemon=[kingambit], active_index=0, is_tera_used=True)
    state = BattleState(p1=p1_side, p2=p2_side)

    evaluator = HybridEvaluator(HeuristicEvaluator(), HeuristicEvaluator(), weight_neural=0.0)
    resolver = SubgameResolver(evaluator=evaluator)

    for depth in (1, 2):
        action, strat, actions, val = resolver.resolve_turn(state, depth=depth)
        assert action.action_type == ActionType.MOVE
        assert action.move_id == "closecombat"
        cc_idx = next(i for i, a in enumerate(actions) if getattr(a, "move_id", "") == "closecombat")
        assert strat[cc_idx] > 0.95


def test_ogerpon_no_spiky_shield_loop(dex):
    """Ogerpon at protect_streak >= 1 must click Ivy Cudgel to finish Kingambit instead of getting trapped in a Spiky Shield loop."""
    ogerpon = Pokemon(
        species="Ogerpon-Wellspring",
        types=(PokemonType.GRASS, PokemonType.WATER),
        current_hp=301,
        max_hp=301,
        raw_stats={"hp": 301, "atk": 279, "def": 204, "spa": 140, "spd": 228, "spe": 350},
        moves=[dex.get_move("ivycudgel"), dex.get_move("hornleech"), dex.get_move("playrough"), dex.get_move("spikyshield")],
        item="wellspringmask",
        protect_streak=1
    )
    dragapult = Pokemon(species="Dragapult", types=(PokemonType.DRAGON, PokemonType.GHOST), current_hp=21, max_hp=317, moves=[dex.get_move("shadowball")])

    kingambit = Pokemon(
        species="Kingambit",
        types=(PokemonType.DARK, PokemonType.STEEL),
        current_hp=129,
        max_hp=341,
        raw_stats={"hp": 341, "atk": 370, "def": 276, "spa": 140, "spd": 206, "spe": 136},
        boosts={"atk": 2},
        moves=[dex.get_move("kowtowcleave"), dex.get_move("suckerpunch"), dex.get_move("ironhead"), dex.get_move("swordsdance")],
    )

    p1_side = BattleSide(pokemon=[ogerpon, dragapult], active_index=0, is_tera_used=True)
    p2_side = BattleSide(pokemon=[kingambit], active_index=0, is_tera_used=True)
    state = BattleState(p1=p1_side, p2=p2_side)

    evaluator = HybridEvaluator(HeuristicEvaluator(), HeuristicEvaluator(), weight_neural=0.0)
    resolver = SubgameResolver(evaluator=evaluator)

    for depth in (1, 2):
        action, strat, actions, val = resolver.resolve_turn(state, depth=depth)
        assert action.action_type == ActionType.MOVE
        assert action.move_id == "ivycudgel"
        ss_idx = next(i for i, a in enumerate(actions) if getattr(a, "move_id", "") == "spikyshield")
        assert strat[ss_idx] < 0.05


def test_gholdengo_defensive_counter_pivot(dex):
    """Gholdengo at -1 SpA with no Air Balloon facing Ground Ting-Lu must switch out to a Ground resist rather than die to Earthquake."""
    gholdengo = Pokemon(
        species="Gholdengo",
        types=(PokemonType.STEEL, PokemonType.GHOST),
        current_hp=158,
        max_hp=315,
        raw_stats={"hp": 315, "atk": 140, "def": 226, "spa": 399, "spd": 218, "spe": 267},
        boosts={"spa": -1},
        moves=[dex.get_move("makeitrain"), dex.get_move("shadowball"), dex.get_move("nastyplot"), dex.get_move("recover")],
        item=None,
        tera_type=PokemonType.FIGHTING
    )
    ogerpon = Pokemon(
        species="Ogerpon-Wellspring",
        types=(PokemonType.GRASS, PokemonType.WATER),
        current_hp=301,
        max_hp=301,
        moves=[dex.get_move("ivycudgel")],
        item="wellspringmask"
    )
    tusk = Pokemon(
        species="Great Tusk",
        types=(PokemonType.GROUND, PokemonType.FIGHTING),
        current_hp=371,
        max_hp=371,
        raw_stats={"hp": 371, "atk": 397, "def": 298, "spa": 127, "spd": 142, "spe": 273},
        moves=[dex.get_move("closecombat")]
    )

    tinglu = Pokemon(
        species="Ting-Lu",
        types=(PokemonType.DARK, PokemonType.GROUND),
        current_hp=385,
        max_hp=514,
        raw_stats={"hp": 514, "atk": 256, "def": 286, "spa": 130, "spd": 295, "spe": 126},
        moves=[dex.get_move("earthquake"), dex.get_move("ruination"), dex.get_move("stealthrock"), dex.get_move("whirlwind")],
        item="leftovers",
        tera_type=PokemonType.POISON
    )

    p1_side = BattleSide(pokemon=[gholdengo, ogerpon, tusk], active_index=0, is_tera_used=True)
    p2_side = BattleSide(pokemon=[tinglu], active_index=0, is_tera_used=False)
    state = BattleState(p1=p1_side, p2=p2_side)

    evaluator = HybridEvaluator(HeuristicEvaluator(), HeuristicEvaluator(), weight_neural=0.0)
    resolver = SubgameResolver(evaluator=evaluator)

    action, strat, actions, val = resolver.resolve_turn(state, depth=1)
    # Gholdengo must switch to Ogerpon (Grass resists Ground) or Great Tusk rather than stay in and click Make It Rain
    assert action.action_type == ActionType.SWITCH


def test_kingambit_defensive_tera_swords_dance(dex):
    """Kingambit facing 2HKO super-effective Earthquake must Terastallize to Flying if clicking Swords Dance."""
    kingambit = Pokemon(
        species="Kingambit",
        types=(PokemonType.DARK, PokemonType.STEEL),
        current_hp=394,
        max_hp=394,
        moves=[dex.get_move("kowtowcleave"), dex.get_move("suckerpunch"), dex.get_move("ironhead"), dex.get_move("swordsdance")],
        item="blackglasses",
        tera_type=PokemonType.FLYING
    )
    tinglu = Pokemon(
        species="Ting-Lu",
        types=(PokemonType.DARK, PokemonType.GROUND),
        current_hp=323,
        max_hp=514,
        raw_stats={"hp": 514, "atk": 256, "def": 286, "spa": 130, "spd": 295, "spe": 126},
        moves=[dex.get_move("earthquake")],
        item="leftovers",
    )

    p1_side = BattleSide(pokemon=[kingambit], active_index=0, is_tera_used=False)
    p2_side = BattleSide(pokemon=[tinglu], active_index=0, is_tera_used=True)
    state = BattleState(p1=p1_side, p2=p2_side)

    evaluator = HybridEvaluator(HeuristicEvaluator(), HeuristicEvaluator(), weight_neural=0.0)
    resolver = SubgameResolver(evaluator=evaluator)

    action, strat, actions, val = resolver.resolve_turn(state, depth=1)
    # If action is swordsdance, it must be Tera Flying
    if getattr(action, "move_id", "") == "swordsdance":
        assert getattr(action, "is_tera", False) is True
        assert getattr(action, "tera_type", None) == PokemonType.FLYING
    # Non-Tera swordsdance must receive 0 probability
    non_tera_sd = next((i for i, a in enumerate(actions) if getattr(a, "move_id", "") == "swordsdance" and not getattr(a, "is_tera", False)), None)
    if non_tera_sd is not None:
        assert strat[non_tera_sd] < 0.05
