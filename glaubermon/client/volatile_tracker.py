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
        self.last_moves = {}
        self.turn = None
        self.acted = set()
        self.switched = set()
        self.paradox_source = {}

    def identify(self, ident):
        return ident[:2], self.identities.get(ident,clean_key(ident.split(':')[-1]))

    def ingest(self, parts):
        if len(parts)>1 and parts[1] == 'upkeep':
            for values in self.mons.values():
                for effect in ('encore','disable','taunt'):
                    value = values.get(effect)
                    if value and value.get('duration',-1)>0:
                        value['duration'] -= 1
                        if not value['duration']:values.pop(effect)
            return
        if len(parts)>2 and parts[1] == 'turn':
            if self.turn != parts[2]:
                self.turn = parts[2]
                self.acted.clear()
                self.switched.clear()
            return
        if len(parts) < 3:
            return
        command,ident = parts[1:3]
        if command in ('switch','drag') and len(parts) > 3:
            key = (ident[:2],clean_key(parts[3].split(',')[0]))
            previous = self.active.get(key[0])
            for mon_key in (previous,key):
                if mon_key:
                    self.mons.pop(mon_key,None)
                    self.last_moves.pop(mon_key,None)
                    self.paradox_source.pop(mon_key,None)
                    for conditions in self.mons.values():
                        for condition in ('trapped','partiallytrapped'):
                            if conditions.get(condition,{}).get('source') == mon_key:
                                conditions.pop(condition,None)
            if self.turn is not None:self.switched.add(key)
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
            self.acted.add(key)
            self.last_move_source = key
            self.last_moves[key] = clean_key(parts[3]) if len(parts)>3 else None
            return
        if command == 'cant':self.acted.add(key)
        if command == 'faint':
            self.mons.pop(key,None)
            return
        if len(parts) < 4:
            return
        effect = clean_key(parts[3].removeprefix('move: ').removeprefix('ability: '))
        values = self.mons.setdefault(key,{})
        if command == '-activate' and effect in ('protosynthesis','quarkdrive'):
            self.paradox_source[key]='[fromitem]' in parts[4:]
        if command == '-start':
            for ability in ('protosynthesis','quarkdrive'):
                stat=effect.removeprefix(ability)
                if effect.startswith(ability) and stat in ('atk','def','spa','spd','spe'):
                    values[ability]={'best_stat':stat,'from_booster':self.paradox_source.pop(key,None)}
            if effect == 'substitute':
                values[effect] = {'hp':-1}
            elif effect in ('encore','disable'):
                move = clean_key(parts[4]) if effect == 'disable' and len(parts)>4 else self.last_moves.get(key)
                duration = -1
                if self.turn is not None:
                    duration = (3 if effect == 'encore' else 4) + int(key in self.acted or key in self.switched)
                    if effect == 'disable' and any('[from] ability:' in part for part in parts[5:]):duration = 4
                values[effect] = {'duration':duration,'move':move}
            elif effect == 'leechseed':
                values[effect] = {'source_tag':'p2' if key[0]=='p1' else 'p1'}
            elif effect == 'taunt':
                values[effect] = {'duration':3+int(key in self.acted or key in self.switched) if self.turn is not None else -1}
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
        if 'leechseed' in values:
            source = values['leechseed'].pop('source_tag',None)
            values['leechseed']['source_side'] = sides[source][0] if source in sides else None
        for condition in ('trapped','partiallytrapped'):
            source = values.get(condition,{}).get('source')
            if source:
                side_idx,side = sides[source[0]]
                index = next((i for i,p in enumerate(side.pokemon) if clean_key(p.species)==source[1]),None)
                values[condition]['source'] = (side_idx,index) if index is not None else None
        return values
