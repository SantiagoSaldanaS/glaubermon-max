"""Unit tests for depth-limited simultaneous subgame search."""

import numpy as np
import pytest
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory
from glaubermon.core.actions import ActionType
from glaubermon.search.subgame_resolver import SubgameResolver


def test_subgame_resolution_and_lethal_detection():
    # P1: Fast Great Tusk with Close Combat (120 BP, Fighting)
    cc = Move.create("Close Combat", PokemonType.FIGHTING, MoveCategory.PHYSICAL, base_power=120)
    tusk = Pokemon(
        species="Great Tusk",
        types=(PokemonType.GROUND, PokemonType.FIGHTING),
        current_hp=350,
        max_hp=350,
        moves=[cc],
        raw_stats={"hp": 350, "atk": 360, "def": 250, "spa": 100, "spd": 100, "spe": 300}
    )

    # P2: Kingambit (4x weak to Fighting, low HP)
    kowtow = Move.create("Kowtow Cleave", PokemonType.DARK, MoveCategory.PHYSICAL, base_power=85)
    kingambit = Pokemon(
        species="Kingambit",
        types=(PokemonType.DARK, PokemonType.STEEL),
        current_hp=100,  # 1 hit from Close Combat will easily KO
        max_hp=400,
        moves=[kowtow],
        raw_stats={"hp": 400, "atk": 380, "def": 250, "spa": 100, "spd": 150, "spe": 120}
    )

    state = BattleState(
        p1=BattleSide(pokemon=[tusk]),
        p2=BattleSide(pokemon=[kingambit])
    )

    resolver = SubgameResolver()
    action, p1_strat, actions, val = resolver.resolve_turn(state, depth=1)

    assert action is not None
    assert np.isclose(np.sum(p1_strat), 1.0)
    assert len(p1_strat) == len(actions)
    # P1 should heavily favor Close Combat to seal the knockout!
    assert action.action_type == ActionType.MOVE
    assert val > 0.5, f"Expected strong winning evaluation, got {val}"
