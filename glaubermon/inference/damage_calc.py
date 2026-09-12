"""Vectorized Gen 9 Damage Calculator & Damage Roll Inversion Engine."""

import math
from typing import List, Optional, Tuple, Any
from glaubermon.core.types import PokemonType, MoveCategory, Weather, Terrain, StatusCondition
from glaubermon.core.constants import get_type_effectiveness, clean_key
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.field_mechanics import effective_weather, poke_round, paradox_environment, best_paradox_stat, resolved_move_type


SLICING_MOVES = {
    "kowtowcleave", "ceaselessedge", "razorshell", "aquacutter", "aerialace",
    "aircutter", "airslash", "crosspoison", "cut", "furycutter", "nightslash",
    "psychocut", "sacredsword", "slash", "solarblade", "stoneaxe", "xscissor",
    "bitterblade", "mightycleave", "tachyoncutter", "psyblade"
}

TYPE_BOOSTING_ITEMS = {
    "blackglasses": PokemonType.DARK,
    "charcoal": PokemonType.FIRE,
    "mysticwater": PokemonType.WATER,
    "miracleseed": PokemonType.GRASS,
    "magnet": PokemonType.ELECTRIC,
    "spelltag": PokemonType.GHOST,
    "sharpbeak": PokemonType.FLYING,
    "silkscarf": PokemonType.NORMAL,
    "softsand": PokemonType.GROUND,
    "hardstone": PokemonType.ROCK,
    "nevermeltice": PokemonType.ICE,
    "dragonfang": PokemonType.DRAGON,
    "poisonbarb": PokemonType.POISON,
    "twistedspoon": PokemonType.PSYCHIC,
    "blackbelt": PokemonType.FIGHTING,
    "metalcoat": PokemonType.STEEL,
}

NON_CONTACT_PHYSICAL = {
    "earthquake", "stoneedge", "rockslide", "rockblast", "iciclespear",
    "seedbomb", "razorleaf", "bonemerang", "bonerush", "bulletseed",
    "pinmissile", "tailslap", "scaleburst", "pyroball", "gunkshot"
}

CONTACT_SPECIAL = {
    "drainingkiss", "grassknot", "infestation", "petaldance", "electrodrift"
}


def is_contact_move(move_id: str, category: MoveCategory = MoveCategory.PHYSICAL) -> bool:
    """Determine whether a move makes physical contact according to canonical Showdown data."""
    try:
        from glaubermon.data.showdown_dex import ShowdownDex
        return ShowdownDex.get_instance().get_move(move_id).is_contact
    except Exception:
        m_clean = clean_key(move_id)
        if category == MoveCategory.PHYSICAL:
            return m_clean not in NON_CONTACT_PHYSICAL
        elif category == MoveCategory.SPECIAL:
            return m_clean in CONTACT_SPECIAL
        return False


NON_REMOVABLE_ITEMS = {
    "wellspringmask", "hearthflamemask", "cornerstonemask",
    "griseousorb", "griseouscore", "rustedsword", "rustedshield"
}


def is_removable_item(item: Optional[str]) -> bool:
    """Check if an item can be removed by Knock Off in Gen 9."""
    if not item:
        return False
    clean = clean_key(item)
    return clean not in NON_REMOVABLE_ITEMS


SHEER_FORCE_MOVES = {
    "ironhead", "flamethrower", "fireblast", "earthpower", "icebeam", "thunderbolt",
    "scald", "rockslide", "playrough", "sludgebomb", "zenheadbutt", "crunch",
    "blizzard", "thunder", "focusblast", "hurricane", "waterfall", "bodyslam",
    "poisonjab", "psychic", "shadowball", "bugbuzz", "flashcannon", "darkpulse"
}


def calculate_damage_rolls(
    attacker: Pokemon,
    defender: Pokemon,
    move: Move,
    weather: Weather = Weather.NONE,
    terrain: Terrain = Terrain.NONE,
    is_critical: bool = False,
    fallen_allies: int = 0,
    attacker_side: Optional[Any] = None,
    defender_side: Optional[Any] = None
) -> List[int]:
    """Calculate all 16 discrete damage rolls for a move in Gen 9."""
    m_id = move.id.lower().replace(" ", "").replace("-", "")

    # Guaranteed critical moves must also be critical in deterministic search.
    if clean_key(defender.ability) in ("battlearmor","shellarmor"):
        is_critical = False
    elif move.will_crit:
        is_critical = True

    weather = effective_weather(weather,attacker,defender)
    # Standalone damage queries may not have a BattleState activation callback.
    if not attacker.booster_stat and paradox_environment(attacker,weather,terrain):
        attacker=attacker.clone();attacker.booster_stat=best_paradox_stat(attacker)
    if not defender.booster_stat and paradox_environment(defender,weather,terrain):
        defender=defender.clone();defender.booster_stat=best_paradox_stat(defender)

    # Fixed damage still respects type immunity.
    if m_id in ("nightshade","seismictoss","superfang"):
        t1,t2 = defender.active_types
        if get_type_effectiveness(move.move_type,t1,t2) == 0:
            return [0]*16
    # 1. Fixed Damage Moves
    if m_id in ("ruination", "superfang"):
        half_hp = max(1, defender.current_hp // 2)
        return [half_hp] * 16
    elif m_id in ("nightshade", "seismictoss"):
        return [max(1, attacker.level)] * 16

    if move.category == MoveCategory.STATUS or move.base_power <= 0:
        return [0] * 16

    # 2. Check protection
    if getattr(defender, "is_protected", False):
        return [0] * 16

    move_type = resolved_move_type(attacker,move,weather,terrain)
    if clean_key(attacker.ability) == "protean" and not attacker.protean_used and not attacker.is_terastallized and m_id != "struggle" and move_type != PokemonType.STELLAR:
        attacker=attacker.clone()
        attacker.type_override=(move_type,None)
        attacker.protean_used=True

    def_ability = (defender.ability or "").lower().replace("-", "").replace(" ", "")
    atk_ability = (attacker.ability or "").lower().replace("-", "").replace(" ", "")
    if not def_ability and defender.species:
        try:
            from glaubermon.data.showdown_dex import ShowdownDex
            def_ability = clean_key(ShowdownDex.get_instance().get_pokemon_ability(defender.species))
        except Exception:
            pass
    if not atk_ability and attacker.species:
        try:
            from glaubermon.data.showdown_dex import ShowdownDex
            atk_ability = clean_key(ShowdownDex.get_instance().get_pokemon_ability(attacker.species))
        except Exception:
            pass
    def_item = (defender.item or "").lower().replace("-", "").replace(" ", "")
    atk_item = (attacker.item or "").lower().replace("-", "").replace(" ", "")

    # 4. Ability & Item Immunities
    # Ground immunities: Levitate, Earth Eater, Air Balloon
    if move_type == PokemonType.GROUND:
        if "airballoon" in def_item or def_ability in ("levitate", "eartheater"):
            return [0] * 16

    # Water immunities: Water Absorb, Storm Drain, Dry Skin
    if move_type == PokemonType.WATER and def_ability in ("waterabsorb", "stormdrain", "dryskin"):
        return [0] * 16

    # Fire immunities: Flash Fire, Well-Baked Body
    if move_type == PokemonType.FIRE and def_ability in ("flashfire", "wellbakedbody"):
        return [0] * 16

    # Electric immunities: Volt Absorb, Lightning Rod, Motor Drive
    if move_type == PokemonType.ELECTRIC and def_ability in ("voltabsorb", "lightningrod", "motordrive"):
        return [0] * 16

    # Grass immunities: Sap Sipper
    if move_type == PokemonType.GRASS and def_ability == "sapsipper":
        return [0] * 16

    # 5. Type effectiveness
    def_types = defender.active_types
    def_t2 = def_types[1] if len(def_types) > 1 else None
    type_mult = get_type_effectiveness(move_type, def_types[0], def_t2)
    if type_mult == 0.0:
        return [0] * 16

    # Wonder Guard: Immune to non-super-effective damage
    if def_ability == "wonderguard" and type_mult <= 1.0 and m_id != "struggle":
        return [0] * 16

    # 6. Determine Attack and Defense stats (Unaware ignores opponent's stat stages)
    ignore_atk_boosts = (def_ability == "unaware")
    ignore_def_boosts = (atk_ability == "unaware")
    if is_critical:
        attack_stat = "def" if m_id == "bodypress" else ("atk" if move.category == MoveCategory.PHYSICAL else "spa")
        defense_stat = "def" if move.category == MoveCategory.PHYSICAL else "spd"
        ignore_atk_boosts |= attacker.boosts.get(attack_stat, 0) < 0
        ignore_def_boosts |= defender.boosts.get(defense_stat, 0) > 0

    offense_mods=[]
    if atk_ability == "flashfire" and "flashfire" in attacker.volatiles and move_type == PokemonType.FIRE:offense_mods.append(6144)
    if move.category == MoveCategory.SPECIAL and def_ability == "vesselofruin" and atk_ability != "vesselofruin":offense_mods.append(3072)
    if move.category == MoveCategory.PHYSICAL:
        offense = attacker
        if m_id == "bodypress":
            # Body Press substitutes Defense's raw stat/stages but still runs
            # Attack modifiers (Band, not Eviolite/Fur Coat/Snow's Defense boost).
            offense = attacker.clone()
            offense.booster_stat = attacker.booster_stat or attacker.get_booster_boosted_stat()
            offense.raw_stats["atk"] = attacker.raw_stats["def"]
            offense.boosts["atk"] = attacker.boosts.get("def",0)
        atk = offense.effective_stat("atk", ignore_boosts=ignore_atk_boosts,extra_mods=offense_mods)
        defense = defender.effective_stat("def", ignore_boosts=ignore_def_boosts)
        if atk_ability in ("hugepower", "purepower"):
            atk = int(atk * 2.0)
        elif atk_ability == "gorillatactics":
            atk = int(atk * 1.5)
        if def_ability == "furcoat":
            defense = int(defense * 2.0)
        # Table of Tablets: Wo-Chien lowers Attack of all other Pokémon by 25%
        if def_ability == "tableoftablets" and atk_ability != "tableoftablets":
            atk = int(atk * 0.75)
        # Sword of Ruin: Chien-Pao lowers Defense of all other Pokémon by 25%
        if (atk_ability == "swordofruin" or def_ability == "swordofruin") and def_ability != "swordofruin":
            defense = int(defense * 0.75)
    else:
        atk = attacker.effective_stat("spa", ignore_boosts=ignore_atk_boosts,extra_mods=offense_mods)
        defense = defender.effective_stat("spd", ignore_boosts=ignore_def_boosts)
        # Beads of Ruin: Chi-Yu lowers Special Defense of all other Pokémon by 25%
        if (atk_ability == "beadsofruin" or def_ability == "beadsofruin") and def_ability != "beadsofruin":
            defense = int(defense * 0.75)

    if move.category == MoveCategory.PHYSICAL and weather == Weather.SNOW and PokemonType.ICE in defender.active_types:
        defense = poke_round(defense*6144)
    if move.category == MoveCategory.SPECIAL and weather == Weather.SANDSTORM and PokemonType.ROCK in defender.active_types:
        defense = poke_round(defense*6144)
    if move.category == MoveCategory.SPECIAL and atk_ability == "solarpower" and weather in (Weather.SUN,Weather.HARSH_SUN):
        atk = poke_round(atk*6144)

    crit_mult = 2.25 if (is_critical and atk_ability == "sniper") else (1.5 if is_critical else 1.0)

    effective_bp = move.base_power
    if m_id == "weatherball" and weather not in (Weather.NONE,Weather.STRONG_WINDS):effective_bp *= 2
    if m_id == "terrainpulse" and terrain != Terrain.NONE and attacker.is_grounded():effective_bp *= 2
    bp_mods=[]
    if atk_ability == "supremeoverlord":
        count=getattr(attacker,"fallen_allies",None)
        if count is None:count=attacker_side.fainted_count if attacker_side is not None else fallen_allies
        bp_mods.append((4096,4506,4915,5325,5734,6144)[min(5,max(0,count))])
    if TYPE_BOOSTING_ITEMS.get(atk_item)==move_type:bp_mods.append(4915)
    if atk_item == "wellspringmask" and clean_key(attacker.species).startswith("ogerponwellspring"):bp_mods.append(4915)
    if atk_ability == "technician" and 0 < effective_bp <= 60:bp_mods.append(6144)
    if m_id == "knockoff" and is_removable_item(defender.item):bp_mods.append(6144)
    if attacker.is_grounded() and {Terrain.GRASSY:PokemonType.GRASS,Terrain.ELECTRIC:PokemonType.ELECTRIC,Terrain.PSYCHIC:PokemonType.PSYCHIC}.get(terrain)==move_type:
        bp_mods.append(5325)
    if defender.is_grounded() and ((terrain==Terrain.GRASSY and m_id in ("earthquake","bulldoze","magnitude")) or (terrain==Terrain.MISTY and move_type==PokemonType.DRAGON)):
        bp_mods.append(2048)
    modifier=4096
    for mod in bp_mods:modifier=(modifier*mod+2048)//4096
    effective_bp=max(1,poke_round(effective_bp*modifier))

    # 7. Base damage formula
    level_factor = int((2 * attacker.level) / 5) + 2
    base_damage = int(int((level_factor * effective_bp * atk) / defense) / 50) + 2

    # 8. Weather modifier
    weather_mult = 1.0
    if weather in (Weather.SUN,Weather.HARSH_SUN):
        if move_type == PokemonType.FIRE:
            weather_mult = 1.5
        elif move_type == PokemonType.WATER:
            # Gen 9 Hydro Steam: power boosted by 1.5x in Sun rather than halved
            if m_id == "hydrosteam":
                weather_mult = 1.5
            else:
                weather_mult = 0.5
    elif weather in (Weather.RAIN,Weather.HEAVY_RAIN):
        if move_type == PokemonType.WATER:
            weather_mult = 1.5
        elif move_type == PokemonType.FIRE:
            weather_mult = 0.5

    if m_id in ("solarbeam", "solarblade") and weather in (Weather.RAIN, Weather.SANDSTORM, Weather.SNOW):
        weather_mult *= 0.5

    # 9. STAB (Same Type Attack Bonus) with Terastallization & Adaptability
    stab_mult = 1.0
    base_types = [t for t in attacker.pretera_types if t is not None]
    has_adaptability = (atk_ability == "adaptability")

    if attacker.is_terastallized and attacker.tera_type:
        is_tera_type_move = (move_type == attacker.tera_type)
        is_base_type_move = (move_type in base_types)

        if is_tera_type_move and is_base_type_move:
            # Tera matches base type: 2.0x STAB (2.25x with Adaptability)
            stab_mult = 2.25 if has_adaptability else 2.0
        elif is_tera_type_move:
            # Tera is new type: 1.5x STAB (2.0x with Adaptability)
            stab_mult = 2.0 if has_adaptability else 1.5
        elif is_base_type_move:
            # Original base type retains 1.5x STAB (2.0x with Adaptability)
            stab_mult = 2.0 if has_adaptability else 1.5
    else:
        if move_type in base_types:
            stab_mult = 2.0 if has_adaptability else 1.5

    # 10. Burn penalty on physical moves (Facade ignores burn drop)
    burn_mult = 1.0
    if attacker.status == StatusCondition.BURN and move.category == MoveCategory.PHYSICAL and m_id != "facade":
        burn_mult = 0.5

    # 11. Ability Damage Multipliers
    ability_mult = 1.0

    # Sharpness: +50% to slicing moves
    if atk_ability == "sharpness" and (getattr(move, "is_slicing", False) or m_id in SLICING_MOVES):
        ability_mult *= 1.5

    # Sheer Force: +30% to moves with secondary effects
    if atk_ability == "sheerforce" and m_id in SHEER_FORCE_MOVES:
        ability_mult *= 1.3

    # Tough Claws: +30% to contact moves
    if atk_ability == "toughclaws" and (getattr(move, "is_contact", False) or is_contact_move(m_id, move.category)):
        ability_mult *= 1.3

    # Water Bubble: 2.0x Water damage dealt, 0.5x Fire damage taken
    if atk_ability == "waterbubble" and move_type == PokemonType.WATER:
        ability_mult *= 2.0
    if def_ability == "waterbubble" and move_type == PokemonType.FIRE:
        ability_mult *= 0.5

    # Tinted Lens: 2.0x damage if resisted
    if atk_ability == "tintedlens" and type_mult < 1.0:
        ability_mult *= 2.0

    # Multiscale / Shadow Shield: 0.5x damage taken when at full HP
    if def_ability in ("multiscale", "shadowshield") and defender.current_hp == defender.max_hp:
        ability_mult *= 0.5

    # Ice Scales: 0.5x Special damage taken
    if def_ability == "icescales" and move.category == MoveCategory.SPECIAL:
        ability_mult *= 0.5

    # Purifying Salt: 0.5x Ghost damage taken
    if def_ability == "purifyingsalt" and move_type == PokemonType.GHOST:
        ability_mult *= 0.5

    # Heatproof: 0.5x Fire damage taken
    if def_ability == "heatproof" and move_type == PokemonType.FIRE:
        ability_mult *= 0.5

    # Fluffy: 0.5x contact damage, 2.0x Fire damage
    if def_ability == "fluffy":
        if getattr(move, "is_contact", False) or is_contact_move(m_id, move.category):
            ability_mult *= 0.5
        if move_type == PokemonType.FIRE:
            ability_mult *= 2.0

    # Solid Rock / Filter / Prism Armor: 0.75x damage taken from super-effective moves
    if def_ability in ("solidrock", "filter", "prismarmor") and type_mult > 1.0:
        ability_mult *= 0.75

    # 12. Item Multipliers
    item_mult = 1.0
    if atk_item:
        if "lifeorb" in atk_item:
            item_mult *= 1.3
        elif "hearthflamemask" in atk_item and move_type == PokemonType.FIRE:
            item_mult *= 1.2
        elif "cornerstonemask" in atk_item and move_type == PokemonType.ROCK:
            item_mult *= 1.2
        elif "expertbelt" in atk_item and type_mult > 1.0:
            item_mult *= 1.2
    # Intermediate calculation before discrete rolls
    mod_damage = base_damage
    mod_damage = int(mod_damage * weather_mult)
    if is_critical:
        mod_damage = int(mod_damage * crit_mult)
    # 13. Gen 9 applies the integer random factor BEFORE STAB/type/burn.
    # Moving it past these steps changes the discrete damage support.
    screen = False
    if defender_side is not None and not is_critical and atk_ability != "infiltrator":
        names = ("auroraveil", "reflect" if move.category == MoveCategory.PHYSICAL else "lightscreen")
        screen = any(defender_side.screens.get(name,0) != 0 for name in names)
    rolls = []
    for percent in range(85, 101):
        final_dmg = mod_damage * percent // 100
        final_dmg = int(final_dmg * stab_mult)
        final_dmg = int(final_dmg * type_mult)
        final_dmg = int(final_dmg * burn_mult)
        final_mod=4096
        # Screen, defensive ability and item modifiers share one fixed-point chain.
        for mod in ([2048] if screen else []) + [round(ability_mult*4096),5324 if atk_item=="lifeorb" else round(item_mult*4096)]:
            final_mod=(final_mod*mod+2048)//4096
        final_dmg = poke_round(final_dmg*final_mod)
        rolls.append(max(1, final_dmg))

    return rolls


def invert_damage_to_stat_bounds(
    observed_damage: int,
    attacker: Pokemon,
    defender: Pokemon,
    move: Move,
    target_stat_is_attacker: bool = True
) -> Tuple[int, int]:
    """Invert an observed damage value to find the exact range of feasible stats.
    
    Returns (min_stat, max_stat) bounding the opponent's unrevealed offensive or defensive stat.
    """
    if observed_damage <= 0 or move.base_power <= 0:
        return (1, 999)

    possible_stats = []
    # Search over possible realistic stat values [50, 600]
    for test_stat in range(50, 601):
        test_atk = attacker.clone()
        test_def = defender.clone()
        if target_stat_is_attacker:
            if move.category == MoveCategory.PHYSICAL:
                test_atk.raw_stats["atk"] = test_stat
            else:
                test_atk.raw_stats["spa"] = test_stat
        else:
            if move.category == MoveCategory.PHYSICAL:
                test_def.raw_stats["def"] = test_stat
            else:
                test_def.raw_stats["spd"] = test_stat

        rolls = calculate_damage_rolls(test_atk, test_def, move)
        if min(rolls) <= observed_damage <= max(rolls):
            possible_stats.append(test_stat)

    if not possible_stats:
        return (50, 600)

    return (min(possible_stats), max(possible_stats))
