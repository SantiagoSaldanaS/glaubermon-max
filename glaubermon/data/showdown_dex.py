"""Showdown Pokédex and Moves Database Cache."""

import os
import re
import json
import requests
from typing import Dict, Tuple, Optional
from glaubermon.core.types import PokemonType, MoveCategory
from glaubermon.core.constants import clean_key


STRING_TO_TYPE: Dict[str, PokemonType] = {
    "normal": PokemonType.NORMAL,
    "fire": PokemonType.FIRE,
    "water": PokemonType.WATER,
    "grass": PokemonType.GRASS,
    "electric": PokemonType.ELECTRIC,
    "ice": PokemonType.ICE,
    "fighting": PokemonType.FIGHTING,
    "poison": PokemonType.POISON,
    "ground": PokemonType.GROUND,
    "flying": PokemonType.FLYING,
    "psychic": PokemonType.PSYCHIC,
    "bug": PokemonType.BUG,
    "rock": PokemonType.ROCK,
    "ghost": PokemonType.GHOST,
    "dragon": PokemonType.DRAGON,
    "steel": PokemonType.STEEL,
    "dark": PokemonType.DARK,
    "fairy": PokemonType.FAIRY,
    "stellar": PokemonType.STELLAR,
}

STRING_TO_CATEGORY: Dict[str, MoveCategory] = {
    "physical": MoveCategory.PHYSICAL,
    "special": MoveCategory.SPECIAL,
    "status": MoveCategory.STATUS,
}

NATURE_MODIFIERS: Dict[str, Tuple[str, str]] = {
    "adamant": ("atk", "spa"),
    "bashful": ("", ""),
    "bold": ("def", "atk"),
    "brave": ("atk", "spe"),
    "calm": ("spd", "atk"),
    "careful": ("spd", "spa"),
    "docile": ("", ""),
    "gentle": ("spd", "def"),
    "hardy": ("", ""),
    "hasty": ("spe", "def"),
    "impish": ("def", "spa"),
    "jolly": ("spe", "spa"),
    "lax": ("def", "spd"),
    "lonely": ("atk", "def"),
    "mild": ("spa", "def"),
    "modest": ("spa", "atk"),
    "naive": ("spe", "spd"),
    "naughty": ("atk", "spd"),
    "quiet": ("spa", "spe"),
    "quirky": ("", ""),
    "rash": ("spa", "spd"),
    "relaxed": ("def", "spe"),
    "sassy": ("spd", "spe"),
    "serious": ("", ""),
    "timid": ("spe", "atk"),
}


class ShowdownDex:
    """Cached offline Showdown Pokédex & Move Database."""

    _instance: Optional["ShowdownDex"] = None

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.pokedex_path = os.path.join(self.data_dir, "pokedex.json")
        self.moves_path = os.path.join(self.data_dir, "moves.json")

        self.pokedex_data: Dict = {}
        self.moves_data: Dict = {}

        self._ensure_data_loaded()

    @classmethod
    def get_instance(cls) -> "ShowdownDex":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_data_loaded(self):
        # 1. Pokedex
        if not os.path.exists(self.pokedex_path):
            print("Downloading Showdown Pokédex database...")
            r = requests.get("https://play.pokemonshowdown.com/data/pokedex.json", timeout=10)
            with open(self.pokedex_path, "w", encoding="utf-8") as f:
                f.write(r.text)
        with open(self.pokedex_path, "r", encoding="utf-8") as f:
            self.pokedex_data = json.load(f)

        # 2. Moves
        if not os.path.exists(self.moves_path):
            print("Downloading Showdown Moves database...")
            r = requests.get("https://play.pokemonshowdown.com/data/moves.json", timeout=10)
            with open(self.moves_path, "w", encoding="utf-8") as f:
                f.write(r.text)
        with open(self.moves_path, "r", encoding="utf-8") as f:
            self.moves_data = json.load(f)

        print(f"ShowdownDex loaded: {len(self.pokedex_data)} Pokémon species, {len(self.moves_data)} moves.")

    def get_pokemon_info(self, species_name: str) -> Tuple[Tuple[PokemonType, Optional[PokemonType]], Dict[str, int]]:
        """Return ((type1, type2), base_stats_dict) for species."""
        key = clean_key(species_name)
        # Handle forme suffixes (e.g. zamazentacrowned -> zamazenta, rotomwash -> rotom)
        entry = self.pokedex_data.get(key)
        if not entry and "-" in species_name:
            base_key = clean_key(species_name.split("-")[0])
            entry = self.pokedex_data.get(base_key)

        if not entry:
            # Fallback
            return (PokemonType.NORMAL, None), {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80}

        types_list = entry.get("types", ["Normal"])
        t1 = STRING_TO_TYPE.get(types_list[0].lower(), PokemonType.NORMAL)
        t2 = STRING_TO_TYPE.get(types_list[1].lower(), None) if len(types_list) > 1 else None
        base_stats = entry.get("baseStats", {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80})
        return (t1, t2), base_stats

    def calculate_pokemon_stats(
        self,
        species_name: str,
        level: int = 100,
        evs: Optional[Dict[str, int]] = None,
        ivs: Optional[Dict[str, int]] = None,
        nature: str = "serious"
    ) -> Dict[str, int]:
        """Compute exact cartridge stats from species base stats, IVs, EVs, and Nature."""
        _, base_stats = self.get_pokemon_info(species_name)
        ev_map = evs or {}
        iv_map = ivs or {}
        nat_plus, nat_minus = NATURE_MODIFIERS.get(clean_key(nature), ("", ""))

        result = {}
        for s in ("hp", "atk", "def", "spa", "spd", "spe"):
            base = base_stats.get(s, 80)
            iv = iv_map.get(s, 31)
            ev = ev_map.get(s, 0)
            if s == "hp":
                if base == 1:  # Shedinja
                    result["hp"] = 1
                else:
                    result["hp"] = int(((2 * base + iv + (ev // 4)) * level) / 100) + level + 10
            else:
                raw = int(((2 * base + iv + (ev // 4)) * level) / 100) + 5
                mult = 1.1 if s == nat_plus else (0.9 if s == nat_minus else 1.0)
                result[s] = int(raw * mult)
        return result

    def get_move(self, move_name_or_id: str) -> "Move":
        """Return an authoritative Move dataclass instance with canonical Showdown attributes."""
        from glaubermon.core.pokemon import Move
        key = clean_key(move_name_or_id)
        entry = self.moves_data.get(key)
        if not entry:
            for k, v in self.moves_data.items():
                if k == key:
                    entry = v
                    break

        if not entry:
            return Move(
                id=key,
                name=move_name_or_id,
                move_type=PokemonType.NORMAL,
                category=MoveCategory.PHYSICAL,
                base_power=80,
                accuracy=1.0,
                priority=0,
                pp=16,
                max_pp=16
            )

        name = entry.get("name", move_name_or_id)
        m_type = STRING_TO_TYPE.get(entry.get("type", "Normal").lower(), PokemonType.NORMAL)
        m_cat = STRING_TO_CATEGORY.get(entry.get("category", "Physical").lower(), MoveCategory.PHYSICAL)
        bp = entry.get("basePower", 0)

        raw_acc = entry.get("accuracy", 100)
        if raw_acc is True:
            acc = 1.0
        elif isinstance(raw_acc, (int, float)):
            acc = max(0.0, min(1.0, raw_acc / 100.0))
        else:
            acc = 1.0

        prio = entry.get("priority", 0)
        base_pp = entry.get("pp", 10)
        max_pp = base_pp if entry.get("noPPBoosts") else int(base_pp * 1.6)

        flags = entry.get("flags", {})
        is_contact = bool(flags.get("contact", 0) == 1)
        is_slicing = bool(flags.get("slicing", 0) == 1)
        is_sound = bool(flags.get("sound", 0) == 1)
        is_reflectable = bool(flags.get("reflectable", 0) == 1)
        is_heal = bool(flags.get("heal", 0) == 1 or entry.get("heal"))
        is_protect = bool(
            entry.get("stallingMove") is True or
            key in ("protect", "detect", "spikyshield", "banefulbunker", "kingsshield",
                    "silktrap", "burningbulwark", "obstruct", "maxguard", "endure")
        )

        drain = None
        d = entry.get("drain")
        if d and isinstance(d, (list, tuple)) and len(d) == 2:
            drain = (int(d[0]), int(d[1]))

        recoil = None
        r = entry.get("recoil")
        if r and isinstance(r, (list, tuple)) and len(r) == 2:
            recoil = (int(r[0]), int(r[1]))

        target = entry.get("target", "normal")
        b = entry.get("boosts")
        sb = entry.get("self", {}).get("boosts") if isinstance(entry.get("self"), dict) else None

        self_b = dict(b) if (target == "self" and b) else {}
        target_b = dict(b) if (target != "self" and b) else {}
        if sb:
            self_b.update(sb)

        sec_list = []
        if isinstance(entry.get("secondaries"), list):
            sec_list.extend(entry["secondaries"])
        elif isinstance(entry.get("secondary"), dict):
            sec_list.append(entry["secondary"])

        return Move(
            id=key,
            name=name,
            move_type=m_type,
            category=m_cat,
            base_power=bp,
            accuracy=acc,
            priority=prio,
            pp=max_pp,
            max_pp=max_pp,
            is_contact=is_contact,
            is_protect=is_protect,
            is_heal=is_heal,
            is_slicing=is_slicing,
            is_sound=is_sound,
            is_reflectable=is_reflectable,
            drain=drain,
            recoil=recoil,
            boosts=target_b if target_b else None,
            self_boosts=self_b if self_b else None,
            target=target,
            always_hits=raw_acc is True,
            crit_ratio=entry.get("critRatio", 1),
            will_crit=bool(entry.get("willCrit", False)),
            blocked_by_protect=bool(flags.get("protect", 0)),
            secondaries=sec_list,
            defrost=bool(flags.get("defrost")),
            sleep_usable=bool(entry.get("sleepUsable")),
            sleep_talk_callable=not (flags.get("nosleeptalk") or flags.get("charge") or entry.get("isZ") or entry.get("isMax")),
        )

    def get_move_info(self, move_name: str) -> Tuple[PokemonType, MoveCategory, int, int, int]:
        """Return (type, category, base_power, accuracy, priority) for move."""
        m = self.get_move(move_name)
        return m.move_type, m.category, m.base_power, int(m.accuracy * 100), m.priority

    def get_pokemon_ability(self, species_name: str) -> str:
        """Authoritatively resolve the canonical competitive ability for ANY Pokémon in Gen 9."""
        key = clean_key(species_name)
        if key in COMPETITIVE_ABILITY_OVERRIDES:
            return COMPETITIVE_ABILITY_OVERRIDES[key]

        entry = self.pokedex_data.get(key)
        if not entry and "-" in species_name:
            base_key = clean_key(species_name.split("-")[0])
            entry = self.pokedex_data.get(base_key)

        if entry and "abilities" in entry:
            abs_dict = entry["abilities"]
            return abs_dict.get("0") or abs_dict.get("H") or abs_dict.get("1") or "Pressure"

        return "Pressure"

    def is_grounded(self, species_name: str, item: Optional[str] = None, ability: Optional[str] = None) -> bool:
        """Check if a Pokémon is affected by Ground moves, Spikes, and Toxic Spikes."""
        (t1, t2), _ = self.get_pokemon_info(species_name)
        if t1 == PokemonType.FLYING or t2 == PokemonType.FLYING:
            return False
        clean_it = clean_key(item) if item else ""
        if clean_it == "airballoon":
            return False
        clean_ab = clean_key(ability) if ability else clean_key(self.get_pokemon_ability(species_name))
        if clean_ab in ("levitate", "eartheater"):
            return False
        return True


COMPETITIVE_ABILITY_OVERRIDES: Dict[str, str] = {
    # High-tier meta threats where Hidden ('H') or secondary ('1') ability is competitive standard
    "dragonite": "Multiscale",
    "gliscor": "Poison Heal",
    "clefable": "Magic Guard",
    "alomomola": "Regenerator",
    "toxapex": "Regenerator",
    "slowking": "Regenerator",
    "slowkinggalar": "Regenerator",
    "tornadustherian": "Regenerator",
    "landorustherian": "Intimidate",
    "landorus": "Sheer Force",
    "cinderace": "Libero",
    "meowscarada": "Protean",
    "serperior": "Contrary",
    "skeledirge": "Unaware",
    "dondozo": "Unaware",
    "clodsire": "Water Absorb",
    "gholdengo": "Good as Gold",
    "kingambit": "Supreme Overlord",
    "heatran": "Flash Fire",
    "orthworm": "Earth Eater",
    "garganacl": "Purifying Salt",
    "corviknight": "Pressure",
    "volcarona": "Flame Body",
    "kyurem": "Pressure",
    "darkrai": "Bad Dreams",
    "rillaboom": "Grassy Surge",
    "weavile": "Pressure",
    "samurotthisui": "Sharpness",
    "gallade": "Sharpness",
    "veluza": "Sharpness",
    "kleavor": "Sharpness",
    "ogerpon": "Defiant",
    "ogerponwellspring": "Water Absorb",
    "ogerponhearthflame": "Mold Breaker",
    "ogerponcornerstone": "Sturdy",
    "terapagos": "Tera Shell",
    "terapagosterastal": "Tera Shell",
    "terapagosstellar": "Teraform Zero",
    "zamazenta": "Dauntless Shield",
    "zamazentacrowned": "Dauntless Shield",
    "zacian": "Intrepid Sword",
    "zaciancrowned": "Intrepid Sword",
    "greattusk": "Protosynthesis",
    "roaringmoon": "Protosynthesis",
    "ironvaliant": "Quark Drive",
    "irontreads": "Quark Drive",
    "ironmoth": "Quark Drive",
    "ironboulder": "Quark Drive",
    "ironcrown": "Quark Drive",
    "ironbundle": "Quark Drive",
    "screamtail": "Protosynthesis",
    "fluttermane": "Protosynthesis",
    "slitherwing": "Protosynthesis",
    "sandyshocks": "Protosynthesis",
    "gougingfire": "Protosynthesis",
    "ragingbolt": "Protosynthesis",
    "walkingwake": "Protosynthesis",
    "pecharunt": "Poison Puppeteer",
    "archaludon": "Stamina",
    "pelipper": "Drizzle",
    "torkoal": "Drought",
    "ninetales": "Drought",
    "ninetalesalola": "Snow Warning",
    "abomasnow": "Snow Warning",
    "politoed": "Drizzle",
    "hippowdon": "Sand Stream",
    "tyranitar": "Sand Stream",
    "tinglu": "Vessel of Ruin",
    "chienpao": "Sword of Ruin",
    "chiyu": "Beads of Ruin",
    "wochien": "Tablets of Ruin",
    "glimmora": "Toxic Debris",
    "hatterene": "Magic Bounce",
    "grimmsnarl": "Prankster",
    "whimsicott": "Prankster",
    "blissey": "Natural Cure",
    "chansey": "Natural Cure",
    "quagsire": "Unaware",
    "rotom": "Levitate",
    "rotomwash": "Levitate",
    "rotomheat": "Levitate",
    "rotomfrost": "Levitate",
    "rotomfan": "Levitate",
    "rotommow": "Levitate",
    "latias": "Levitate",
    "latios": "Levitate",
    "cresselia": "Levitate",
    "hydreigon": "Levitate",
    "weezing": "Levitate",
    "weezinggalar": "Levitate",
    "flygon": "Levitate",
    "eelektross": "Levitate",
    "bronzong": "Levitate",
    "mismagius": "Levitate",
    "claydol": "Levitate",
    "cryogonal": "Levitate",
    "sinistcha": "Hospitality",
    "polteageist": "Weak Armor",
    "ursalfuna": "Guts",
    "ursalunabloodmoon": "Mind's Eye",
    "enamorus": "Contrary",
    "enamorustherian": "Overcoat",
    "deoxys": "Pressure",
    "deoxysspeed": "Pressure",
    "deoxysdefense": "Pressure",
    "deoxysattack": "Pressure",
}
