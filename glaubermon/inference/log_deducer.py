"""Showdown protocol event stream parser for strategic log deduction."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from glaubermon.core.types import Hazard


@dataclass
class OpponentDeductions:
    """Inferred constraints on an opponent's Pokémon."""
    species: str
    revealed_moves: Set[str] = field(default_factory=set)
    revealed_item: Optional[str] = None
    excluded_items: Set[str] = field(default_factory=set)
    revealed_ability: Optional[str] = None
    min_speed: int = 0
    max_speed: int = 999
    is_choice_locked: bool = False
    choice_locked_move: Optional[str] = None


class LogDeducer:
    """Parses real-time battle messages and updates deduction states."""

    def __init__(self):
        self.deductions: Dict[str, OpponentDeductions] = {}
        self.last_move_turn: Optional[int] = None
        self.moves_this_turn: List[Tuple[str, str, int]] = []  # (player_id, species, priority)

    def get_or_create(self, species: str) -> OpponentDeductions:
        clean = species.strip().lower()
        if clean not in self.deductions:
            self.deductions[clean] = OpponentDeductions(species=clean)
        return self.deductions[clean]

    def parse_line(self, line: str, current_turn: int = 1):
        """Parse a single line from Pokémon Showdown protocol."""
        parts = line.strip().split("|")
        if len(parts) < 2:
            return

        cmd = parts[1]

        # 1. Damage from hazards (Heavy-Duty Boots check)
        if cmd == "-damage":
            # |-damage|p2a: Dragonite|75/100|[from] Stealth Rock
            if len(parts) >= 5 and "Stealth Rock" in parts[4]:
                species = self._extract_species(parts[2])
                if species:
                    ded = self.get_or_create(species)
                    ded.excluded_items.add("heavydutyboots")

        # 2. Item reveal / loss
        elif cmd == "-item":
            # |-item|p2a: Great Tusk|Booster Energy|[from] ability: Protosynthesis
            if len(parts) >= 4:
                species = self._extract_species(parts[2])
                item_name = parts[3].lower().replace(" ", "").replace("-", "")
                if species:
                    ded = self.get_or_create(species)
                    ded.revealed_item = item_name

        # 3. Ability reveal
        elif cmd == "-ability":
            # |-ability|p2a: Dondozo|Unaware
            if len(parts) >= 4:
                species = self._extract_species(parts[2])
                ability_name = parts[3].lower().replace(" ", "")
                if species:
                    ded = self.get_or_create(species)
                    ded.revealed_ability = ability_name

        # 4. Move reveals & Choice lock tracking
        elif cmd == "move":
            # |move|p2a: Kingambit|Sucker Punch|p1a: Dragapult
            if len(parts) >= 4:
                species = self._extract_species(parts[2])
                move_id = parts[3].lower().replace(" ", "").replace("-", "")
                if species:
                    ded = self.get_or_create(species)
                    ded.revealed_moves.add(move_id)
                    # If previously used another move while active without switching
                    if ded.choice_locked_move and ded.choice_locked_move != move_id:
                        ded.excluded_items.add("choicescarf")
                        ded.excluded_items.add("choiceband")
                        ded.excluded_items.add("choicespecs")
                    ded.choice_locked_move = move_id

        # 5. Switch resets choice lock
        elif cmd == "switch":
            # |switch|p2a: Kingambit|...
            if len(parts) >= 3:
                species = self._extract_species(parts[2])
                if species:
                    ded = self.get_or_create(species)
                    ded.choice_locked_move = None

    def update_speed_bounds(self, faster_species: str, slower_species: str, slower_speed: int):
        """Update speed lower bound if faster_species moved first on equal priority."""
        ded = self.get_or_create(faster_species)
        ded.min_speed = max(ded.min_speed, slower_speed)

    def _extract_species(self, ident: str) -> Optional[str]:
        # ident format: "p1a: Dragapult" or "p2a: Great Tusk"
        tokens = ident.split(":")
        if len(tokens) >= 2:
            return tokens[1].strip().lower()
        return None
