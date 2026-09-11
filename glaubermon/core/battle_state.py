"""Full 6v6 Battle State Representation."""

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Dict, List, Optional
from glaubermon.core.types import Weather, Terrain, Hazard, ActionType, MoveCategory, PokemonType
from glaubermon.core.pokemon import Pokemon
from glaubermon.core.actions import Action, MoveAction, SwitchAction
from glaubermon.core.constants import clean_key


@dataclass
class BattleSide:
    """State of one side of the battlefield (player or opponent)."""
    pokemon: List[Pokemon] = field(default_factory=list)
    active_index: int = 0
    hazards: Dict[Hazard, int] = field(default_factory=dict)
    screens: Dict[str, int] = field(default_factory=dict)  # "reflect", "lightscreen", "auroraveil"
    tailwind: int = 0
    is_tera_used: bool = False

    @property
    def active_pokemon(self) -> Optional[Pokemon]:
        if 0 <= self.active_index < len(self.pokemon):
            return self.pokemon[self.active_index]
        return None

    @property
    def fainted_count(self) -> int:
        return sum(1 for p in self.pokemon if p.is_fainted)

    @property
    def is_all_fainted(self) -> bool:
        return len(self.pokemon) > 0 and self.fainted_count == len(self.pokemon)

    def available_switches(self) -> List[int]:
        """Indices of Pokémon on the bench that are alive and can be switched into."""
        switches = []
        for i, p in enumerate(self.pokemon):
            if i != self.active_index and not p.is_fainted:
                switches.append(i)
        return switches

    def clone(self) -> "BattleSide":
        return BattleSide(
            pokemon=[p.clone() for p in self.pokemon],
            active_index=self.active_index,
            hazards=dict(self.hazards),
            screens=dict(self.screens),
            tailwind=self.tailwind,
            is_tera_used=self.is_tera_used
        )


@dataclass
class BattleState:
    """Complete 6v6 competitive battle state."""
    p1: BattleSide
    p2: BattleSide
    weather: Weather = Weather.NONE
    weather_turns: int = 0
    terrain: Terrain = Terrain.NONE
    terrain_turns: int = 0
    turn: int = 1
    trick_room: int = 0
    pending_switches: tuple = ()
    continuation: Optional[dict] = None

    def is_trapped(self, player: int) -> bool:
        side = self.p1 if player == 1 else self.p2
        mon = side.active_pokemon
        if not mon or mon.is_fainted or PokemonType.GHOST in mon.active_types or clean_key(mon.item) == "shedshell":
            return False
        return any(key in mon.volatiles for key in ("partiallytrapped", "trapped", "request_trapped"))

    @property
    def is_game_over(self) -> bool:
        return self.p1.is_all_fainted or self.p2.is_all_fainted

    @property
    def winner(self) -> Optional[int]:
        if self.p1.is_all_fainted and not self.p2.is_all_fainted:
            return 2
        elif self.p2.is_all_fainted and not self.p1.is_all_fainted:
            return 1
        return None

    def get_valid_actions(self, player: int = 1) -> List[Action]:
        """Generate all legal actions for the specified player (1 or 2)."""
        side = self.p1 if player == 1 else self.p2
        active = side.active_pokemon
        actions: List[Action] = []

        if self.pending_switches and player not in self.pending_switches:
            return []  # This player is waiting, not selecting another move.
        if player in self.pending_switches or active is None or active.is_fainted:
            # Must switch
            for slot in side.available_switches():
                actions.append(SwitchAction(target_slot=slot + 1, species=side.pokemon[slot].species))
            return actions

        # 1. Move actions
        it = clean_key(active.item)
        is_choice = it in ("choicespecs", "choiceband", "choicescarf")
        locked_m = active.choice_locked_move if is_choice else None

        for i, move in enumerate(active.moves):
            if move.pp > 0:
                if "taunt" in active.volatiles and move.category == MoveCategory.STATUS and move.id != "mefirst":
                    continue
                if locked_m and clean_key(move.id) != clean_key(locked_m):
                    continue
                actions.append(MoveAction(
                    move_id=move.id,
                    move_slot=i + 1,
                    is_tera=False
                ))
                # Add Terastallization option if not yet used
                if not side.is_tera_used and active.tera_type is not None and not active.is_terastallized:
                    actions.append(MoveAction(
                        move_id=move.id,
                        move_slot=i + 1,
                        is_tera=True,
                        tera_type=active.tera_type
                    ))

        # Exhausted/Choice-locked moves still allow Struggle, even with a bench.
        if not actions:
            actions.append(MoveAction(move_id="struggle", move_slot=1))

        # 2. Switch actions
        for slot in ([] if self.is_trapped(player) else side.available_switches()):
            actions.append(SwitchAction(
                target_slot=slot + 1,
                species=side.pokemon[slot].species
            ))

        return actions

    def flipped(self) -> "BattleState":
        """Change player perspective, including source-bound and queued state."""
        result = self.clone()
        result.p1,result.p2 = result.p2,result.p1
        result.pending_switches = tuple(sorted(3-i for i in result.pending_switches))
        for side in (result.p1,result.p2):
            for mon in side.pokemon:
                for key in ("trapped","partiallytrapped"):
                    source = mon.volatiles.get(key,{}).get("source")
                    if source:
                        mon.volatiles[key]["source"] = (3-source[0],source[1])
        continuation = result.continuation
        if continuation:
            flip = lambda player: "p2" if player == "p1" else "p1"
            continuation["users"] = {flip(who):index for who,index in continuation["users"].items()}
            continuation["order"] = [(flip(who),slot,move) for who,slot,move in continuation["order"]]
            continuation["moved"] = [flip(who) for who in continuation["moved"]]
            continuation["flinched"] = [flip(who) for who in continuation["flinched"]]
            continuation["m1"],continuation["m2"] = continuation["m2"],continuation["m1"]
        return result

    def clone(self) -> "BattleState":
        return BattleState(
            p1=self.p1.clone(),
            p2=self.p2.clone(),
            weather=self.weather,
            weather_turns=self.weather_turns,
            terrain=self.terrain,
            terrain_turns=self.terrain_turns,
            turn=self.turn,
            trick_room=self.trick_room,
            pending_switches=tuple(self.pending_switches),
            continuation=deepcopy(self.continuation),
        )
