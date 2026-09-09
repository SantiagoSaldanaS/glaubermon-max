"""Constants, Type Chart, and Mathematical Formulas for Pokémon Mechanics."""

import re
from typing import Optional, Tuple
from glaubermon.core.types import PokemonType, StatusCondition, Weather, Terrain


def clean_key(name: Optional[str]) -> str:
    """Authoritative Showdown ID normalizer (lowercase alphanumeric only)."""
    if not name:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(name)).lower()

# 18-type effectiveness chart (Attacking Type -> Defending Type -> Multiplier)
# Defaults to 1.0 if not listed.
TYPE_CHART = {
    PokemonType.NORMAL: {
        PokemonType.ROCK: 0.5, PokemonType.GHOST: 0.0, PokemonType.STEEL: 0.5
    },
    PokemonType.FIRE: {
        PokemonType.FIRE: 0.5, PokemonType.WATER: 0.5, PokemonType.GRASS: 2.0,
        PokemonType.ICE: 2.0, PokemonType.BUG: 2.0, PokemonType.ROCK: 0.5,
        PokemonType.DRAGON: 0.5, PokemonType.STEEL: 2.0
    },
    PokemonType.WATER: {
        PokemonType.FIRE: 2.0, PokemonType.WATER: 0.5, PokemonType.GRASS: 0.5,
        PokemonType.GROUND: 2.0, PokemonType.ROCK: 2.0, PokemonType.DRAGON: 0.5
    },
    PokemonType.GRASS: {
        PokemonType.FIRE: 0.5, PokemonType.WATER: 2.0, PokemonType.GRASS: 0.5,
        PokemonType.POISON: 0.5, PokemonType.GROUND: 2.0, PokemonType.FLYING: 0.5,
        PokemonType.BUG: 0.5, PokemonType.ROCK: 2.0, PokemonType.DRAGON: 0.5,
        PokemonType.STEEL: 0.5
    },
    PokemonType.ELECTRIC: {
        PokemonType.WATER: 2.0, PokemonType.ELECTRIC: 0.5, PokemonType.GRASS: 0.5,
        PokemonType.GROUND: 0.0, PokemonType.FLYING: 2.0, PokemonType.DRAGON: 0.5
    },
    PokemonType.ICE: {
        PokemonType.FIRE: 0.5, PokemonType.WATER: 0.5, PokemonType.GRASS: 2.0,
        PokemonType.ICE: 0.5, PokemonType.GROUND: 2.0, PokemonType.FLYING: 2.0,
        PokemonType.DRAGON: 2.0, PokemonType.STEEL: 0.5
    },
    PokemonType.FIGHTING: {
        PokemonType.NORMAL: 2.0, PokemonType.ICE: 2.0, PokemonType.POISON: 0.5,
        PokemonType.FLYING: 0.5, PokemonType.PSYCHIC: 0.5, PokemonType.BUG: 0.5,
        PokemonType.ROCK: 2.0, PokemonType.GHOST: 0.0, PokemonType.DARK: 2.0,
        PokemonType.STEEL: 2.0, PokemonType.FAIRY: 0.5
    },
    PokemonType.POISON: {
        PokemonType.GRASS: 2.0, PokemonType.POISON: 0.5, PokemonType.GROUND: 0.5,
        PokemonType.ROCK: 0.5, PokemonType.GHOST: 0.5, PokemonType.STEEL: 0.0,
        PokemonType.FAIRY: 2.0
    },
    PokemonType.GROUND: {
        PokemonType.FIRE: 2.0, PokemonType.ELECTRIC: 2.0, PokemonType.GRASS: 0.5,
        PokemonType.POISON: 2.0, PokemonType.FLYING: 0.0, PokemonType.BUG: 0.5,
        PokemonType.ROCK: 2.0, PokemonType.STEEL: 2.0
    },
    PokemonType.FLYING: {
        PokemonType.ELECTRIC: 0.5, PokemonType.GRASS: 2.0, PokemonType.FIGHTING: 2.0,
        PokemonType.BUG: 2.0, PokemonType.ROCK: 0.5, PokemonType.STEEL: 0.5
    },
    PokemonType.PSYCHIC: {
        PokemonType.FIGHTING: 2.0, PokemonType.POISON: 2.0, PokemonType.PSYCHIC: 0.5,
        PokemonType.DARK: 0.0, PokemonType.STEEL: 0.5
    },
    PokemonType.BUG: {
        PokemonType.FIRE: 0.5, PokemonType.GRASS: 2.0, PokemonType.FIGHTING: 0.5,
        PokemonType.POISON: 0.5, PokemonType.FLYING: 0.5, PokemonType.PSYCHIC: 2.0,
        PokemonType.GHOST: 0.5, PokemonType.DARK: 2.0, PokemonType.STEEL: 0.5,
        PokemonType.FAIRY: 0.5
    },
    PokemonType.ROCK: {
        PokemonType.FIRE: 2.0, PokemonType.ICE: 2.0, PokemonType.FIGHTING: 0.5,
        PokemonType.GROUND: 0.5, PokemonType.FLYING: 2.0, PokemonType.BUG: 2.0,
        PokemonType.STEEL: 0.5
    },
    PokemonType.GHOST: {
        PokemonType.NORMAL: 0.0, PokemonType.PSYCHIC: 2.0, PokemonType.GHOST: 2.0,
        PokemonType.DARK: 0.5
    },
    PokemonType.DRAGON: {
        PokemonType.DRAGON: 2.0, PokemonType.STEEL: 0.5, PokemonType.FAIRY: 0.0
    },
    PokemonType.STEEL: {
        PokemonType.FIRE: 0.5, PokemonType.WATER: 0.5, PokemonType.ELECTRIC: 0.5,
        PokemonType.ICE: 2.0, PokemonType.ROCK: 2.0, PokemonType.STEEL: 0.5,
        PokemonType.FAIRY: 2.0
    },
    PokemonType.DARK: {
        PokemonType.FIGHTING: 0.5, PokemonType.PSYCHIC: 2.0, PokemonType.GHOST: 2.0,
        PokemonType.DARK: 0.5, PokemonType.FAIRY: 0.5
    },
    PokemonType.FAIRY: {
        PokemonType.FIRE: 0.5, PokemonType.FIGHTING: 2.0, PokemonType.POISON: 0.5,
        PokemonType.DRAGON: 2.0, PokemonType.DARK: 2.0, PokemonType.STEEL: 0.5
    },
}


def get_type_effectiveness(
    attacking: PokemonType,
    defending1: PokemonType,
    defending2: Optional[PokemonType] = None
) -> float:
    """Calculate the cumulative type effectiveness multiplier."""
    if attacking in (PokemonType.UNKNOWN, PokemonType.STELLAR):
        # Stellar deals 2.0x against Tera, 1.2x on STAB, baseline 1.0x
        return 1.0

    mult = TYPE_CHART.get(attacking, {}).get(defending1, 1.0)
    if defending2 and defending2 not in (PokemonType.UNKNOWN, PokemonType.STELLAR):
        mult *= TYPE_CHART.get(attacking, {}).get(defending2, 1.0)
    return mult


# Stat boost stage multipliers (-6 to +6)
STAT_STAGE_MULTIPLIERS = {
    -6: 2.0 / 8.0,
    -5: 2.0 / 7.0,
    -4: 2.0 / 6.0,
    -3: 2.0 / 5.0,
    -2: 2.0 / 4.0,
    -1: 2.0 / 3.0,
    0: 1.0,
    1: 3.0 / 2.0,
    2: 4.0 / 2.0,
    3: 5.0 / 2.0,
    4: 6.0 / 2.0,
    5: 7.0 / 2.0,
    6: 8.0 / 2.0,
}


def compute_stat(
    base: int,
    iv: int = 31,
    ev: int = 0,
    level: int = 100,
    nature_mult: float = 1.0,
    is_hp: bool = False
) -> int:
    """Compute exact cartridge stat from base, IV, EV, and Nature."""
    if is_hp:
        if base == 1:  # Shedinja
            return 1
        return int(((2 * base + iv + (ev // 4)) * level) / 100) + level + 10
    else:
        raw = int(((2 * base + iv + (ev // 4)) * level) / 100) + 5
        return int(raw * nature_mult)


# Standard 16 discrete damage rolls in Pokémon
DAMAGE_ROLLS = [
    0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92,
    0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00
]
