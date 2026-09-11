"""Public field state: expiration comes from protocol, never hidden item guesses."""
from dataclasses import dataclass, field
from glaubermon.core.constants import clean_key


@dataclass
class PublicFieldTracker:
    turn: int = 0
    screens: dict = field(default_factory=lambda: {'p1':{},'p2':{}})
    tailwind: dict = field(default_factory=lambda: {'p1':0,'p2':0})
    trick_room: int = 0

    def ingest(self, parts):
        command = parts[1]
        if command == 'turn' and len(parts)>2:
            turn = int(parts[2])
            elapsed = max(0, turn-self.turn)
            self.turn = turn
            self.trick_room = max(0,self.trick_room-elapsed)
            for side in self.tailwind:
                self.tailwind[side] = max(0,self.tailwind[side]-elapsed)
            # Screens can last 5 or 8 turns depending on a hidden item. -1 means
            # publicly active, remaining duration unknown. Only -sideend clears it.
        elif command in ('-sidestart','-sideend') and len(parts)>3:
            side, effect = parts[2][:2], clean_key(parts[3].removeprefix('move: '))
            if side not in self.screens:
                return
            if effect in ('reflect','lightscreen','auroraveil'):
                if command == '-sidestart':
                    self.screens[side][effect] = -1
                else:
                    self.screens[side].pop(effect,None)
            elif effect == 'tailwind':
                self.tailwind[side] = 4 if command == '-sidestart' else 0
        elif command in ('-fieldstart','-fieldend') and len(parts)>2:
            if clean_key(parts[2].removeprefix('move: ')) == 'trickroom':
                self.trick_room = 5 if command == '-fieldstart' else 0
