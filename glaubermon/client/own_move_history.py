"""Private move requests plus public PP expenditure; scoped to one battle.

A one-entry Struggle request is an action menu, not a replacement moveset.
Unknown PP remain estimates explicitly marked as such.
"""
from glaubermon.core.constants import clean_key


class OwnMoveHistory:
    def __init__(self, dex):
        self.dex = dex
        self.moves = {}
        self.abilities = {}
        self.active = {}
        self.complete_start = False

    @staticmethod
    def key(ident):
        return ident[:2], clean_key(ident.split(':', 1)[-1])

    def ingest(self, parts):
        if len(parts) < 2:
            return
        command = parts[1]
        if command == 'start':
            self.complete_start = True
        if len(parts) < 3:
            return
        key = self.key(parts[2])
        if command in ('switch', 'drag'):
            self.active[key[0]] = key
        elif command == '-ability' and len(parts) > 3:
            self.abilities[key] = clean_key(parts[3])
        elif command == 'cant' and len(parts) > 4 and parts[3] == 'nopp':
            move = self.moves.get(key, {}).get(clean_key(parts[4]))
            if move:
                move.pp = 0
                move.pp_known = True
        elif command == 'move' and len(parts) > 3:
            # Sleep Talk/Metronome/Dancer spend the caller's PP, not the called move's.
            if any(part.startswith('[from] ') for part in parts[5:]):
                return
            move = self.moves.get(key, {}).get(clean_key(parts[3]))
            if move is None or move.id == 'struggle':
                return
            cost = 1
            if move.target in ('normal','allAdjacentFoes','allAdjacent','any','randomNormal','foeSide'):
                other = self.active.get('p2' if key[0] == 'p1' else 'p1')
                ability = self.abilities.get(other)
                if ability == 'pressure':
                    cost += 1
                elif ability is None and not self.complete_start:
                    move.pp_known = False
            move.pp = max(0, move.pp - cost)

    def observe(self, pokemon, active_moves=None):
        """Return independent Move objects in actual move-slot order."""
        key = self.key(pokemon.get('ident', pokemon.get('details', '')))
        stored = self.moves.setdefault(key, {})
        struggle_only = bool(active_moves) and all(clean_key(m.get('id','')) == 'struggle' for m in active_moves)
        menu = None if struggle_only else active_moves
        entries = menu if menu else pokemon.get('moves', [])
        result = []
        for entry in entries:
            move_id = clean_key(entry.get('id','')) if isinstance(entry,dict) else clean_key(entry)
            move = stored.get(move_id)
            if move is None:
                move = self.dex.get_move(move_id)
                move.pp_known = False
            else:
                move = move.clone()
            move.request_disabled = struggle_only
            if isinstance(entry,dict):
                if 'pp' in entry:
                    move.pp = int(entry['pp'])
                    move.pp_known = True
                if 'maxpp' in entry:
                    move.max_pp = int(entry['maxpp'])
                move.request_disabled = bool(entry.get('disabled',False))
            stored[move_id] = move.clone()
            result.append(move)
        # A changed moveset must not resurrect an older transformed/copied slot.
        self.moves[key] = {move.id:move.clone() for move in result}
        return result
