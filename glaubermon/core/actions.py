"""Typed representation of battle actions in Pokémon Showdown."""

from dataclasses import dataclass
from typing import Optional
from glaubermon.core.types import ActionType, PokemonType


@dataclass(frozen=True)
class Action:
    """Base class for player actions."""
    action_type: ActionType

    def to_showdown_command(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class MoveAction(Action):
    """Action representing choosing a move."""
    move_id: str
    move_slot: int  # 1 to 4
    target: int = 0  # 0 for default/single, 1-2 for specific doubles target
    is_tera: bool = False
    tera_type: Optional[PokemonType] = None

    def __init__(
        self,
        move_id: str,
        move_slot: int,
        target: int = 0,
        is_tera: bool = False,
        tera_type: Optional[PokemonType] = None
    ):
        object.__setattr__(self, "action_type", ActionType.MOVE)
        object.__setattr__(self, "move_id", move_id.lower().replace(" ", "").replace("-", ""))
        object.__setattr__(self, "move_slot", move_slot)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "is_tera", is_tera)
        object.__setattr__(self, "tera_type", tera_type)

    def to_showdown_command(self) -> str:
        cmd = f"/choose move {self.move_slot}"
        if self.is_tera:
            cmd += " terastallize"
        return cmd


@dataclass(frozen=True)
class SwitchAction(Action):
    """Action representing switching to a bench Pokémon."""
    target_slot: int  # 1 to 6 (slot in party)
    species: str

    def __init__(self, target_slot: int, species: str):
        object.__setattr__(self, "action_type", ActionType.SWITCH)
        object.__setattr__(self, "target_slot", target_slot)
        object.__setattr__(self, "species", species)

    def to_showdown_command(self) -> str:
        return f"/choose switch {self.target_slot}"


def action_to_logit_index(action: Action) -> int:
    """Map typed Action to the neural network's 14-dim output logit index:
    - Indices 0..3: Regular moves (slots 1..4)
    - Indices 4..7: Terastallized moves (slots 1..4)
    - Indices 8..13: Switches (party slots 1..6)
    """
    if action.action_type == ActionType.MOVE:
        slot = getattr(action, "move_slot", 1) - 1  # 0 to 3
        is_tera = getattr(action, "is_tera", False)
        return (slot + 4) if is_tera else min(max(0, slot), 3)
    elif action.action_type == ActionType.SWITCH:
        target = getattr(action, "target_slot", 1) - 1  # 0 to 5
        return 8 + min(max(0, target), 5)
    return 0
