"""Official Smogon Gen 9 OU Tournament Meta Sample Teams.

Contains standard tournament-winning 6v6 archetypes:
1. Standard Balance (Tusk + Gholdengo + Kingambit Core)
2. Hyper Offense (Booster Energy + Hazard Pressure)
3. Bulky Hazard Stall (Dondozo + Gliscor + Corviknight)
"""

from typing import Dict, List, Optional
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, Hazard
from glaubermon.data.showdown_dex import ShowdownDex, STRING_TO_TYPE


from glaubermon.core.constants import clean_key


STANDARD_ABILITIES: Dict[str, str] = {
    "greattusk": "Protosynthesis",
    "gholdengo": "Good as Gold",
    "kingambit": "Supreme Overlord",
    "dragapult": "Infiltrator",
    "ogerponwellspring": "Water Absorb",
    "tinglu": "Vessel of Ruin",
    "ironvaliant": "Quark Drive",
    "roaringmoon": "Protosynthesis",
    "samurotthisui": "Sharpness",
    "dondozo": "Unaware",
    "gliscor": "Poison Heal",
    "corviknight": "Pressure",
    "heatran": "Flash Fire",
    "dragonite": "Multiscale",
    "zamazenta": "Dauntless Shield",
    "rillaboom": "Grassy Surge",
    "landorustherian": "Intimidate",
    "kyurem": "Pressure",
    "darkrai": "Bad Dreams",
    "weavile": "Pressure",
    "clefable": "Magic Guard",
    "enamorus": "Cute Charm",
    "walkingwake": "Protosynthesis",
    "gougingfire": "Protosynthesis",
    "primarina": "Torrent",
    "meowscarada": "Protean",
    "hatterene": "Magic Bounce",
    "clodsire": "Water Absorb",
    "alomomola": "Regenerator",
    "skeledirge": "Unaware",
    "cinderace": "Libero",
    "ironboulder": "Quark Drive",
    "ironmoth": "Quark Drive",
    "ironcrown": "Quark Drive",
    "archaludon": "Stamina",
    "pelipper": "Drizzle",
    "barraskewda": "Swift Swim",
}


STANDARD_ITEMS: Dict[str, str] = {
    "greattusk": "Booster Energy",
    "gholdengo": "Air Balloon",
    "kingambit": "Black Glasses",
    "dragapult": "Choice Specs",
    "ogerponwellspring": "Wellspring Mask",
    "tinglu": "Leftovers",
    "ironvaliant": "Booster Energy",
    "roaringmoon": "Booster Energy",
    "samurotthisui": "Focus Sash",
    "dondozo": "Leftovers",
    "gliscor": "Toxic Orb",
    "corviknight": "Leftovers",
    "heatran": "Leftovers",
    "dragonite": "Heavy-Duty Boots",
    "zamazenta": "Leftovers",
    "rillaboom": "Choice Band",
    "landorustherian": "Rocky Helmet",
    "kyurem": "Loaded Dice",
    "darkrai": "Life Orb",
    "weavile": "Heavy-Duty Boots",
    "clefable": "Leftovers",
    "enamorus": "Choice Scarf",
    "walkingwake": "Booster Energy",
    "gougingfire": "Booster Energy",
    "primarina": "Leftovers",
    "meowscarada": "Choice Band",
    "hatterene": "Leftovers",
    "clodsire": "Heavy-Duty Boots",
    "alomomola": "Heavy-Duty Boots",
    "skeledirge": "Heavy-Duty Boots",
    "cinderace": "Heavy-Duty Boots",
    "ironboulder": "Booster Energy",
    "ironmoth": "Booster Energy",
    "ironcrown": "Booster Energy",
    "archaludon": "Assault Vest",
    "pelipper": "Damp Rock",
    "barraskewda": "Choice Band",
}


STANDARD_MOVESETS: Dict[str, List[str]] = {
    "greattusk": ["Close Combat", "Headlong Rush", "Ice Spinner", "Rapid Spin"],
    "gholdengo": ["Make It Rain", "Shadow Ball", "Nasty Plot", "Recover"],
    "kingambit": ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Swords Dance"],
    "dragapult": ["Draco Meteor", "Shadow Ball", "Flamethrower", "U-turn"],
    "ogerponwellspring": ["Ivy Cudgel", "Horn Leech", "Play Rough", "Spiky Shield"],
    "tinglu": ["Earthquake", "Ruination", "Stealth Rock", "Whirlwind"],
    "ironvaliant": ["Moonblast", "Close Combat", "Knock Off", "Thunderbolt"],
    "roaringmoon": ["Knock Off", "Dragon Dance", "Earthquake", "Acrobatics"],
    "samurotthisui": ["Ceaseless Edge", "Razor Shell", "Knock Off", "Sucker Punch"],
    "dondozo": ["Liquidation", "Body Press", "Rest", "Sleep Talk"],
    "gliscor": ["Earthquake", "Toxic", "Protect", "Spikes"],
    "corviknight": ["Brave Bird", "Body Press", "Roost", "Defog"],
    "heatran": ["Magma Storm", "Earth Power", "Flash Cannon", "Stealth Rock"],
    "dragonite": ["Extreme Speed", "Earthquake", "Dragon Dance", "Roost"],
    "zamazenta": ["Close Combat", "Crunch", "Iron Defense", "Body Press"],
    "rillaboom": ["Grassy Glide", "Wood Hammer", "Knock Off", "U-turn"],
    "landorustherian": ["Earthquake", "U-turn", "Stealth Rock", "Taunt"],
    "kyurem": ["Freeze-Dry", "Ice Beam", "Earth Power", "Scale Shot"],
    "darkrai": ["Dark Pulse", "Sludge Bomb", "Ice Beam", "Focus Blast"],
    "weavile": ["Knock Off", "Triple Axel", "Ice Shard", "Swords Dance"],
    "clefable": ["Moonblast", "Soft-Boiled", "Stealth Rock", "Calm Mind"],
    "enamorus": ["Moonblast", "Earth Power", "Mystical Fire", "Healing Wish"],
    "walkingwake": ["Hydro Steam", "Draco Meteor", "Flamethrower", "Dragon Pulse"],
    "gougingfire": ["Dragon Dance", "Heat Crash", "Earthquake", "Outrage"],
    "primarina": ["Moonblast", "Hydro Pump", "Psychic Noise", "Calm Mind"],
    "meowscarada": ["Flower Trick", "Knock Off", "U-turn", "Triple Axel"],
    "hatterene": ["Dazzling Gleam", "Psyshock", "Mystical Fire", "Calm Mind"],
    "clodsire": ["Earthquake", "Toxic", "Recover", "Stealth Rock"],
    "alomomola": ["Flip Turn", "Wish", "Protect", "Scald"],
    "skeledirge": ["Torch Song", "Shadow Ball", "Slack Off", "Will-O-Wisp"],
    "cinderace": ["Pyro Ball", "Court Change", "U-turn", "Gunk Shot"],
    "ironboulder": ["Mighty Cleave", "Zen Headbutt", "Close Combat", "Swords Dance"],
    "ironmoth": ["Fiery Dance", "Sludge Wave", "Energy Ball", "Psychic"],
    "ironcrown": ["Tachyon Cutter", "Psyshock", "Focus Blast", "Volt Switch"],
    "archaludon": ["Electro Shot", "Draco Meteor", "Flash Cannon", "Body Press"],
    "pelipper": ["Hurricane", "Hydro Pump", "U-turn", "Roost"],
    "barraskewda": ["Liquidation", "Close Combat", "Aqua Jet", "Flip Turn"],
}


STANDARD_EV_PRESETS: Dict[str, str] = {
    "greattusk": "fast_phys",
    "gholdengo": "fast_spec",
    "kingambit": "physical",
    "dragapult": "fast_spec",
    "ogerponwellspring": "fast_phys",
    "tinglu": "bulky",
    "ironvaliant": "fast_spec",
    "roaringmoon": "fast_phys",
    "samurotthisui": "physical",
    "dondozo": "bulky",
    "gliscor": "bulky",
    "corviknight": "bulky",
    "heatran": "special",
    "dragonite": "physical",
    "zamazenta": "fast_phys",
    "rillaboom": "physical",
    "landorustherian": "bulky",
    "kyurem": "special",
    "darkrai": "fast_spec",
    "weavile": "fast_phys",
    "clefable": "bulky",
    "enamorus": "fast_spec",
    "walkingwake": "fast_spec",
    "gougingfire": "fast_phys",
    "primarina": "special",
    "meowscarada": "fast_phys",
    "hatterene": "special",
    "clodsire": "bulky",
    "alomomola": "bulky",
    "skeledirge": "bulky",
    "cinderace": "fast_phys",
    "ironboulder": "fast_phys",
    "ironmoth": "fast_spec",
    "ironcrown": "fast_spec",
    "archaludon": "bulky",
    "pelipper": "bulky",
    "barraskewda": "fast_phys",
}


STANDARD_TERA_TYPES: Dict[str, str] = {
    "greattusk": "Ice",
    "gholdengo": "Fighting",
    "kingambit": "Flying",
    "dragapult": "Ghost",
    "ogerponwellspring": "Water",
    "tinglu": "Poison",
    "ironvaliant": "Fairy",
    "roaringmoon": "Flying",
    "samurotthisui": "Ghost",
    "dondozo": "Grass",
    "gliscor": "Water",
    "corviknight": "Dragon",
    "heatran": "Grass",
    "dragonite": "Normal",
    "zamazenta": "Fire",
    "rillaboom": "Grass",
    "landorustherian": "Water",
    "kyurem": "Ice",
    "darkrai": "Poison",
    "weavile": "Ice",
    "clefable": "Steel",
    "enamorus": "Fairy",
    "walkingwake": "Water",
    "gougingfire": "Fire",
    "primarina": "Steel",
    "meowscarada": "Dark",
    "hatterene": "Water",
    "clodsire": "Steel",
    "alomomola": "Grass",
    "skeledirge": "Fairy",
    "cinderace": "Fire",
    "ironboulder": "Fighting",
    "ironmoth": "Grass",
    "ironcrown": "Fighting",
    "archaludon": "Electric",
    "pelipper": "Ground",
    "barraskewda": "Water",
}


def get_meta_pokemon_by_species(species: str) -> Optional[Pokemon]:
    """Retrieve or construct canonical competitive meta Pokémon for a species."""
    clean = clean_key(species)
    # 1. Check STANDARD_MOVESETS dictionary first
    if clean in STANDARD_MOVESETS:
        moves = STANDARD_MOVESETS[clean]
        item = STANDARD_ITEMS.get(clean, "Leftovers")
        ab = ShowdownDex.get_instance().get_pokemon_ability(species)
        ev = STANDARD_EV_PRESETS.get(clean, "balanced")
        tera = STANDARD_TERA_TYPES.get(clean, None)
        return build_meta_pokemon(species, moves, item=item, tera_type_str=tera, ev_preset=ev, ability=ab)

    # 2. Check all pre-built archetypes
    for arch_fn in (get_meta_team_balance, get_meta_team_hyper_offense, get_meta_team_stall):
        for mon in arch_fn():
            if clean_key(mon.species) == clean:
                return mon

    # 3. Construct on-the-fly meta Pokémon from ShowdownDex for any of the 1,517 species
    dex = ShowdownDex.get_instance()
    entry = dex.pokedex_data.get(clean)
    if not entry and "-" in species:
        entry = dex.pokedex_data.get(clean_key(species.split("-")[0]))
    if entry:
        ab = dex.get_pokemon_ability(species)
        item = STANDARD_ITEMS.get(clean, "Leftovers")
        tera = STANDARD_TERA_TYPES.get(clean, None)
        return build_meta_pokemon(species, [], item=item, tera_type_str=tera, ability=ab)

    return None


def build_meta_pokemon(
    species: str,
    move_names: List[str],
    item: Optional[str] = None,
    tera_type_str: Optional[object] = None,
    ev_preset: str = "balanced",
    ability: Optional[str] = None
) -> Pokemon:
    """Construct a fully competitive Pokémon with exact stats, moves, item, and ability."""
    dex = ShowdownDex.get_instance()
    types, base_stats = dex.get_pokemon_info(species)
    clean_spec = species.lower().replace("-", "").replace(" ", "")

    # Resolve competitive ability and item
    if ability is None:
        ability = dex.get_pokemon_ability(species)
    if item is None:
        item = STANDARD_ITEMS.get(clean_spec, "Leftovers")

    # Standard Level 100 31-IV competitive stats
    base_hp = base_stats.get("hp", 80)
    base_atk = base_stats.get("atk", 80)
    base_def = base_stats.get("def", 80)
    base_spa = base_stats.get("spa", 80)
    base_spd = base_stats.get("spd", 80)
    base_spe = base_stats.get("spe", 80)

    # Calculate actual Lv 100 stats with 252/252 competitive EV spreads
    max_hp = base_hp * 2 + 141
    if ev_preset == "bulky":
        max_hp += 63  # 252 EVs in HP

    calc_stats = {
        "hp": max_hp,
        "atk": base_atk * 2 + 36 + (63 if ev_preset in ("physical", "fast_phys") else 0),
        "def": base_def * 2 + 36 + (63 if ev_preset == "bulky" else 0),
        "spa": base_spa * 2 + 36 + (63 if ev_preset in ("special", "fast_spec") else 0),
        "spd": base_spd * 2 + 36 + (63 if ev_preset == "bulky" else 0),
        "spe": base_spe * 2 + 36 + (63 if ev_preset in ("fast_phys", "fast_spec") else 0),
    }

    moves = []
    for m_name in move_names:
        m_type, m_cat, bp, acc, prio = dex.get_move_info(m_name)
        if m_name.lower().replace(" ", "").replace("-", "") == "ivycudgel":
            if species == "Ogerpon-Wellspring" or "wellspring" in item.lower():
                m_type = PokemonType.WATER
            elif species == "Ogerpon-Hearthflame" or "hearthflame" in item.lower():
                m_type = PokemonType.FIRE
            elif species == "Ogerpon-Cornerstone" or "cornerstone" in item.lower():
                m_type = PokemonType.ROCK
        moves.append(Move.create(m_name, m_type, m_cat, base_power=bp, accuracy=acc / 100.0, priority=prio))

    if isinstance(tera_type_str, PokemonType):
        tera_type = tera_type_str
    elif isinstance(tera_type_str, str):
        tera_type = STRING_TO_TYPE.get(tera_type_str.lower(), types[0])
    else:
        tera_type = types[0]

    return Pokemon(
        species=species,
        types=types,
        max_hp=max_hp,
        current_hp=max_hp,
        moves=moves,
        raw_stats=calc_stats,
        item=item,
        ability=ability,
        tera_type=tera_type
    )


def get_meta_team_balance() -> List[Pokemon]:
    """Standard Gen 9 OU Balance (The #1 Tournament Archetype)."""
    return [
        build_meta_pokemon("Great Tusk", ["Close Combat", "Headlong Rush", "Ice Spinner", "Rapid Spin"], "Booster Energy", "Ice", "fast_phys"),
        build_meta_pokemon("Gholdengo", ["Make It Rain", "Shadow Ball", "Nasty Plot", "Recover"], "Air Balloon", "Fighting", "fast_spec"),
        build_meta_pokemon("Kingambit", ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Swords Dance"], "Black Glasses", "Flying", "physical"),
        build_meta_pokemon("Dragapult", ["Draco Meteor", "Shadow Ball", "Flamethrower", "U-turn"], "Choice Specs", "Ghost", "fast_spec"),
        build_meta_pokemon("Ogerpon-Wellspring", ["Ivy Cudgel", "Horn Leech", "Play Rough", "Spiky Shield"], "Wellspring Mask", "Water", "fast_phys"),
        build_meta_pokemon("Ting-Lu", ["Earthquake", "Ruination", "Stealth Rock", "Whirlwind"], "Leftovers", "Poison", "bulky"),
    ]


def get_meta_team_hyper_offense() -> List[Pokemon]:
    """Standard Gen 9 OU Hyper Offense (High Ladder Sweepers)."""
    return [
        build_meta_pokemon("Iron Valiant", ["Moonblast", "Close Combat", "Knock Off", "Thunderbolt"], "Booster Energy", "Fairy", "fast_spec"),
        build_meta_pokemon("Roaring Moon", ["Knock Off", "Dragon Dance", "Earthquake", "Acrobatics"], "Booster Energy", "Flying", "fast_phys"),
        build_meta_pokemon("Samurott-Hisui", ["Ceaseless Edge", "Razor Shell", "Knock Off", "Sucker Punch"], "Focus Sash", "Ghost", "physical"),
        build_meta_pokemon("Great Tusk", ["Headlong Rush", "Close Combat", "Ice Spinner", "Rapid Spin"], "Heavy-Duty Boots", "Ice", "fast_phys"),
        build_meta_pokemon("Kingambit", ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Swords Dance"], "Black Glasses", "Dark", "physical"),
        build_meta_pokemon("Gholdengo", ["Make It Rain", "Shadow Ball", "Focus Blast", "Nasty Plot"], "Choice Scarf", "Steel", "fast_spec"),
    ]


def get_meta_team_stall() -> List[Pokemon]:
    """Standard Gen 9 OU Bulky Stall / Hazard Control."""
    return [
        build_meta_pokemon("Dondozo", ["Liquidation", "Body Press", "Rest", "Sleep Talk"], "Leftovers", "Grass", "bulky"),
        build_meta_pokemon("Gliscor", ["Earthquake", "Toxic", "Protect", "Spikes"], "Toxic Orb", "Water", "bulky"),
        build_meta_pokemon("Corviknight", ["Brave Bird", "Body Press", "Roost", "Defog"], "Leftovers", "Dragon", "bulky"),
        build_meta_pokemon("Heatran", ["Magma Storm", "Earth Power", "Flash Cannon", "Stealth Rock"], "Leftovers", "Grass", "special"),
        build_meta_pokemon("Great Tusk", ["Rapid Spin", "Knock Off", "Earthquake", "Ice Spinner"], "Heavy-Duty Boots", "Water", "bulky"),
        build_meta_pokemon("Kingambit", ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Swords Dance"], "Leftovers", "Flying", "physical"),
    ]


def get_meta_team_pelol94() -> List[Pokemon]:
    """Pelol94's exact competitive team from live user Showdown battles."""
    return [
        build_meta_pokemon("Iron Valiant", ["Moonblast", "Close Combat", "Knock Off", "Thunderbolt"], "Booster Energy", "Fairy", "fast_spec"),
        build_meta_pokemon("Gliscor", ["Earthquake", "Toxic", "Protect", "Spikes"], "Toxic Orb", "Water", "bulky"),
        build_meta_pokemon("Great Tusk", ["Close Combat", "Headlong Rush", "Ice Spinner", "Rapid Spin"], "Booster Energy", "Ice", "fast_phys"),
        build_meta_pokemon("Ting-Lu", ["Earthquake", "Ruination", "Stealth Rock", "Whirlwind"], "Leftovers", "Poison", "bulky"),
        build_meta_pokemon("Dragapult", ["Draco Meteor", "Shadow Ball", "Flamethrower", "U-turn"], "Choice Specs", "Ghost", "fast_spec"),
        build_meta_pokemon("Kingambit", ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Swords Dance"], "Black Glasses", "Flying", "physical"),
    ]


META_TEAMS: Dict[str, callable] = {
    "balance": get_meta_team_balance,
    "hyper_offense": get_meta_team_hyper_offense,
    "stall": get_meta_team_stall,
    "pelol94": get_meta_team_pelol94,
}


def create_meta_battle(p1_archetype: str = "balance", p2_archetype: str = "balance") -> BattleState:
    """Create a 6v6 tournament meta battle between two competitive archetypes."""
    p1_mons = META_TEAMS.get(p1_archetype, get_meta_team_balance)()
    p2_mons = META_TEAMS.get(p2_archetype, get_meta_team_balance)()

    return BattleState(
        p1=BattleSide(pokemon=p1_mons, hazards={Hazard.STEALTH_ROCK: 0}),
        p2=BattleSide(pokemon=p2_mons, hazards={Hazard.STEALTH_ROCK: 0})
    )
