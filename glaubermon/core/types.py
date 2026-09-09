"""Core enumeration types for Pokémon mechanics."""

from enum import Enum, auto


class PokemonType(Enum):
    NORMAL = "Normal"
    FIRE = "Fire"
    WATER = "Water"
    GRASS = "Grass"
    ELECTRIC = "Electric"
    ICE = "Ice"
    FIGHTING = "Fighting"
    POISON = "Poison"
    GROUND = "Ground"
    FLYING = "Flying"
    PSYCHIC = "Psychic"
    BUG = "Bug"
    ROCK = "Rock"
    GHOST = "Ghost"
    DRAGON = "Dragon"
    STEEL = "Steel"
    DARK = "Dark"
    FAIRY = "Fairy"
    STELLAR = "Stellar"
    UNKNOWN = "???"

    @classmethod
    def from_str(cls, val: str) -> "PokemonType":
        normalized = val.strip().capitalize()
        for member in cls:
            if member.value.lower() == normalized.lower():
                return member
        return cls.UNKNOWN


class MoveCategory(Enum):
    PHYSICAL = "Physical"
    SPECIAL = "Special"
    STATUS = "Status"


class StatusCondition(Enum):
    NONE = "none"
    BURN = "brn"
    PARALYSIS = "par"
    POISON = "psn"
    TOXIC = "tox"
    SLEEP = "slp"
    FREEZE = "frz"


class Weather(Enum):
    NONE = "none"
    SUN = "sun"
    RAIN = "rain"
    SANDSTORM = "sandstorm"
    SNOW = "snow"
    HARSH_SUN = "desolateland"
    HEAVY_RAIN = "primordialsea"
    STRONG_WINDS = "deltastream"


class Terrain(Enum):
    NONE = "none"
    ELECTRIC = "electric"
    GRASSY = "grassy"
    MISTY = "misty"
    PSYCHIC = "psychic"


class Hazard(Enum):
    STEALTH_ROCK = "stealthrock"
    SPIKES_1 = "spikes_1"
    SPIKES_2 = "spikes_2"
    SPIKES_3 = "spikes_3"
    TOXIC_SPIKES_1 = "toxicspikes_1"
    TOXIC_SPIKES_2 = "toxicspikes_2"
    STICKY_WEB = "stickyweb"


class ActionType(Enum):
    MOVE = auto()
    SWITCH = auto()
