"""Track revealed volatile conditions without inventing hidden HP or timers."""
from copy import deepcopy
from glaubermon.core.constants import clean_key

BIND_MOVES = {'bind','clamp','firespin','infestation','magmastorm','sandtomb','snaptrap','thundercage','whirlpool','wrap'}


class PublicVolatileTracker:
    def __init__(self):
        self.mons = {}
        self.active = {}
        self.identities = {}
        self.last_move_source = None
        self.entry_once = {}

    def identify(self, ident):
        return ident[:2], self.identities.get(ident,clean_key(ident.split(':')[-1]))

    def ingest(self, parts):
        if len(parts) < 3:
            return
        command,ident = parts[1:3]
        if command in ('switch','drag') and len(parts) > 3:
            key = (ident[:2],clean_key(parts[3].split(',')[0]))
            previous = self.active.get(key[0])
            for mon_key in (previous,key):
                if mon_key:
                    self.mons.pop(mon_key,None)
                    for conditions in self.mons.values():
                        for condition in ('trapped','partiallytrapped'):
                            if conditions.get(condition,{}).get('source') == mon_key:
                                conditions.pop(condition,None)
            self.active[key[0]] = key
            self.identities[ident] = key[1]
            return
        key = self.identify(ident)
        if key[0] not in ('p1','p2'):
            return
        if command == '-boost':
            for flag,name in (('shield_boosted','Dauntless Shield'),('sword_boosted','Intrepid Sword')):
                if any(name in part for part in parts[4:]):
                    self.entry_once.setdefault(key,set()).add(flag)
        if command == 'move':
            self.last_move_source = key
            return
        if command == 'faint':
            self.mons.pop(key,None)
            return
        if len(parts) < 4:
            return
        effect = clean_key(parts[3].removeprefix('move: '))
        values = self.mons.setdefault(key,{})
        if command == '-start':
            if effect == 'substitute':
                values[effect] = {'hp':-1}
            elif effect == 'taunt':
                values[effect] = {'duration':-1}
            elif effect == 'confusion':
                values[effect] = {'time':-1}
        elif command == '-activate' and (effect in BIND_MOVES or effect == 'trapped'):
            source = next((self.identify(p[5:]) for p in parts[4:] if p.startswith('[of] ')), self.last_move_source)
            condition = 'partiallytrapped' if effect in BIND_MOVES else 'trapped'
            values[condition] = {'source':source}
            if condition == 'partiallytrapped':
                values[condition].update(duration=-1,divisor=8)
        elif command == '-end':
            values.pop('partiallytrapped' if effect in BIND_MOVES else effect,None)

    def observations(self, tag, species, sides):
        values = deepcopy(self.mons.get((tag,clean_key(species)),{}))
        for condition in ('trapped','partiallytrapped'):
            source = values.get(condition,{}).get('source')
            if source:
                side_idx,side = sides[source[0]]
                index = next((i for i,p in enumerate(side.pokemon) if clean_key(p.species)==source[1]),None)
                values[condition]['source'] = (side_idx,index) if index is not None else None
        return values
