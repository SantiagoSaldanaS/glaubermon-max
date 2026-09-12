"""Gen 9 field rules shared by search transitions and damage calculations."""
from glaubermon.core.types import Weather,Terrain,PokemonType,StatusCondition,MoveCategory
from glaubermon.core.constants import clean_key

WEATHER_MOVES={'sunnyday':Weather.SUN,'raindance':Weather.RAIN,'sandstorm':Weather.SANDSTORM,'snowscape':Weather.SNOW}
WEATHER_ABILITIES={'drought':Weather.SUN,'drizzle':Weather.RAIN,'sandstream':Weather.SANDSTORM,'snowwarning':Weather.SNOW}
TERRAIN_MOVES={'electricterrain':Terrain.ELECTRIC,'grassyterrain':Terrain.GRASSY,'psychicterrain':Terrain.PSYCHIC,'mistyterrain':Terrain.MISTY}
TERRAIN_ABILITIES={'electricsurge':Terrain.ELECTRIC,'grassysurge':Terrain.GRASSY,'psychicsurge':Terrain.PSYCHIC,'mistysurge':Terrain.MISTY}


def poke_round(numerator,denominator=4096):
    """Showdown fixed-point rounding: exact halves round down."""
    return (2*numerator+denominator-1)//(2*denominator)


def effective_weather(weather,*mons):
    return Weather.NONE if any(mon and not mon.is_fainted and clean_key(mon.ability) in ('cloudnine','airlock') for mon in mons) else weather


def best_paradox_stat(mon):
    def value(stat):
        raw,stage=mon.raw_stats.get(stat,100),mon.boosts.get(stat,0)
        return raw*(2+stage)//2 if stage>=0 else raw*2//(2-stage)
    return max(('atk','def','spa','spd','spe'),key=value)


def paradox_environment(mon,weather,terrain):
    ability=clean_key(mon.ability)
    return (ability=='protosynthesis' and weather==Weather.SUN) or (ability=='quarkdrive' and terrain==Terrain.ELECTRIC)


def sync_paradox(state):
    weather=effective_weather(state.weather,state.p1.active_pokemon,state.p2.active_pokemon)
    for side in (state.p1,state.p2):
        mon=side.active_pokemon
        if not mon or mon.is_fainted:continue
        ability=clean_key(mon.ability)
        if ability not in ('protosynthesis','quarkdrive'):continue
        effect=mon.volatiles.get(ability)
        active=paradox_environment(mon,weather,state.terrain)
        if effect and effect.get('from_booster') is False and not active:
            mon.volatiles.pop(ability)
            mon.booster_stat=None
            effect=None
        if effect:
            mon.booster_stat=effect['best_stat']
            continue
        if active:
            mon.booster_stat=best_paradox_stat(mon)
            mon.volatiles[ability]={'best_stat':mon.booster_stat,'from_booster':False}
        elif clean_key(mon.item)=='boosterenergy':
            mon.booster_stat=best_paradox_stat(mon)
            mon.item=None
            mon.volatiles[ability]={'best_stat':mon.booster_stat,'from_booster':True}
        elif mon.booster_stat:
            # Legacy/public state can reveal the stat without revealing its source.
            mon.volatiles[ability]={'best_stat':mon.booster_stat,'from_booster':None}


def set_weather(state,weather,source):
    if state.weather == weather:return
    state.weather=weather
    rock={Weather.SUN:'heatrock',Weather.RAIN:'damprock',Weather.SANDSTORM:'smoothrock',Weather.SNOW:'icyrock'}.get(weather)
    state.weather_turns=8 if clean_key(source.item)==rock else 5
    sync_paradox(state)


def set_terrain(state,terrain,source):
    if state.terrain == terrain:return
    state.terrain=terrain
    state.terrain_turns=8 if clean_key(source.item)=='terrainextender' else 5
    sync_paradox(state)


def effective_speed(mon,side,weather,terrain):
    stage=mon.boosts.get('spe',0)
    speed=mon.raw_stats.get('spe',100)
    speed=speed*(2+stage)//2 if stage>=0 else speed*2//(2-stage)
    ability,item=clean_key(mon.ability),clean_key(mon.item)
    mods=[]
    if ((ability=='swiftswim' and weather in (Weather.RAIN,Weather.HEAVY_RAIN)) or
        (ability=='chlorophyll' and weather in (Weather.SUN,Weather.HARSH_SUN)) or
        (ability=='sandrush' and weather==Weather.SANDSTORM) or
        (ability=='slushrush' and weather==Weather.SNOW) or
        (ability=='surgesurfer' and terrain==Terrain.ELECTRIC)):
        mods.append(8192)
    if ability=='quickfeet' and mon.status!=StatusCondition.NONE:mods.append(6144)
    boosted = mon.booster_stat or mon.get_booster_boosted_stat()
    if boosted is None and paradox_environment(mon,weather,terrain):boosted = best_paradox_stat(mon)
    if boosted == 'spe':mods.append(6144)
    if item=='choicescarf':mods.append(6144)
    if item in ('ironball','machobrace','poweranklet','powerband','powerbelt','powerbracer','powerlens','powerweight'):mods.append(2048)
    if side is not None and side.tailwind:mods.append(8192)
    modifier=4096
    for mod in mods:modifier=(modifier*mod+2048)//4096
    speed=poke_round(speed*modifier)
    if mon.status==StatusCondition.PARALYSIS and ability!='quickfeet':speed//=2
    return max(1,min(10000,speed))


def move_priority(mon,move):
    priority=move.priority
    if clean_key(mon.ability)=='prankster' and move.category==MoveCategory.STATUS:priority+=1
    if clean_key(mon.ability)=='galewings' and move.move_type==PokemonType.FLYING and mon.current_hp==mon.max_hp:priority+=1
    return priority


def accuracy_chance(attacker,defender,move,weather):
    if move.always_hits or clean_key(attacker.ability)=='noguard' or clean_key(defender.ability)=='noguard':return 1.0
    if move.id in ('thunder','hurricane'):
        if weather in (Weather.RAIN,Weather.HEAVY_RAIN):return 1.0
        accuracy=.5 if weather in (Weather.SUN,Weather.HARSH_SUN) else move.accuracy
    elif move.id=='blizzard' and weather==Weather.SNOW:return 1.0
    else:accuracy=move.accuracy
    a=max(-6,min(6,attacker.boosts.get('accuracy',0)))
    d=max(-6,min(6,defender.boosts.get('evasion',0)))
    stage=max(-6,min(6,a-d))
    modifier=(3+stage)/3 if stage>=0 else 3/(3-stage)
    if clean_key(attacker.item)=='widelens':modifier*=1.1
    if clean_key(attacker.ability)=='compoundeyes':modifier*=1.3
    return min(1.0,max(0.0,int(accuracy*100*modifier)/100))


def weather_residual(state):
    # Weather expires at residual order 1, before dealing chip on that turn.
    if state.weather_turns>0:
        state.weather_turns-=1
        if state.weather_turns==0:state.weather=Weather.NONE
    sync_paradox(state)
    weather=effective_weather(state.weather,state.p1.active_pokemon,state.p2.active_pokemon)
    for side in (state.p1,state.p2):
        mon=side.active_pokemon
        if not mon or mon.is_fainted:continue
        ability=clean_key(mon.ability)
        if weather==Weather.SANDSTORM and not any(t in mon.active_types for t in (PokemonType.ROCK,PokemonType.GROUND,PokemonType.STEEL)):
            if ability not in ('magicguard','overcoat','sandveil','sandrush','sandforce') and clean_key(mon.item)!='safetygoggles':
                mon.take_damage(max(1,mon.max_hp//16))
        if mon.is_fainted:continue
        if weather in (Weather.RAIN,Weather.HEAVY_RAIN):
            if ability=='raindish':mon.heal(max(1,mon.max_hp//16))
            if ability=='dryskin':mon.heal(max(1,mon.max_hp//8))
        if weather in (Weather.SUN,Weather.HARSH_SUN) and ability in ('dryskin','solarpower'):
            mon.take_damage(max(1,mon.max_hp//8))
        if weather==Weather.SNOW and ability=='icebody':mon.heal(max(1,mon.max_hp//16))


def hit_count(attacker,move,rng,sampled):
    hits=move.multihit
    if not hits:return 1
    if isinstance(hits,(list,tuple)):
        if clean_key(attacker.ability)=='skilllink':return hits[1]
        if clean_key(attacker.item)=='loadeddice' and tuple(hits)==(2,5):
            return rng.choice((4,5)) if sampled else 5
        if tuple(hits)==(2,5):return rng.choice((2,)*7+(3,)*7+(4,)*3+(5,)*3) if sampled else 3
        return rng.randint(*hits) if sampled else sum(hits)//2
    if hits==10 and clean_key(attacker.item)=='loadeddice':return rng.randint(4,10) if sampled else 7
    return hits
