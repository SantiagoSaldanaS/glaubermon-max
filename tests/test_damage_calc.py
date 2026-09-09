"""Unit tests for Gen 9 damage calculator and roll inversion."""

import pytest
from glaubermon.core.types import PokemonType, MoveCategory, Weather, StatusCondition
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.inference.damage_calc import calculate_damage_rolls, invert_damage_to_stat_bounds


def test_type_effectiveness_and_immunity():
    # Ghost vs Normal = 0 damage
    ghost_mon = Pokemon(species="Gengar", types=(PokemonType.GHOST, PokemonType.POISON))
    normal_mon = Pokemon(species="Snorlax", types=(PokemonType.NORMAL, None))
    shadow_ball = Move.create("Shadow Ball", PokemonType.GHOST, MoveCategory.SPECIAL, base_power=80)

    rolls = calculate_damage_rolls(ghost_mon, normal_mon, shadow_ball)
    assert all(r == 0 for r in rolls), "Ghost move should deal 0 damage to Normal type"


def test_super_effective_damage_rolls():
    # Close Combat (Fighting, 120 BP) vs Kingambit (Dark/Steel - 4x weak to Fighting)
    tusk = Pokemon(
        species="Great Tusk",
        types=(PokemonType.GROUND, PokemonType.FIGHTING),
        raw_stats={"hp": 371, "atk": 361, "def": 298, "spa": 127, "spd": 142, "spe": 273}
    )
    kingambit = Pokemon(
        species="Kingambit",
        types=(PokemonType.DARK, PokemonType.STEEL),
        raw_stats={"hp": 404, "atk": 405, "def": 276, "spa": 140, "spd": 206, "spe": 136}
    )
    close_combat = Move.create("Close Combat", PokemonType.FIGHTING, MoveCategory.PHYSICAL, base_power=120)

    rolls = calculate_damage_rolls(tusk, kingambit, close_combat)
    assert len(rolls) == 16
    assert min(rolls) > 300, f"Expected massive 4x damage, got min {min(rolls)}"
    # Verify monotonic 16 rolls
    assert all(rolls[i] <= rolls[i + 1] for i in range(15))


def test_reverse_damage_inversion():
    # Test that an observed damage can be inverted to find the correct attack stat range
    attacker = Pokemon(
        species="Garchomp",
        types=(PokemonType.DRAGON, PokemonType.GROUND),
        raw_stats={"hp": 357, "atk": 359, "def": 226, "spa": 176, "spd": 206, "spe": 303}
    )
    defender = Pokemon(
        species="Heatran",
        types=(PokemonType.FIRE, PokemonType.STEEL),
        raw_stats={"hp": 323, "atk": 194, "def": 248, "spa": 296, "spd": 248, "spe": 190}
    )
    earthquake = Move.create("Earthquake", PokemonType.GROUND, MoveCategory.PHYSICAL, base_power=100)

    rolls = calculate_damage_rolls(attacker, defender, earthquake)
    observed_damage = rolls[8]  # Median roll

    stat_min, stat_max = invert_damage_to_stat_bounds(
        observed_damage, attacker, defender, earthquake, target_stat_is_attacker=True
    )
    assert stat_min <= 359 <= stat_max, f"True stat 359 not in inverted range [{stat_min}, {stat_max}]"
