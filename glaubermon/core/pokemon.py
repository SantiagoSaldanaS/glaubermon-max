"""Detailed state representation for individual Pokémon and Moves."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from glaubermon.core.types import PokemonType, MoveCategory, StatusCondition, Hazard
from glaubermon.core.constants import STAT_STAGE_MULTIPLIERS, compute_stat, clean_key, get_type_effectiveness


@dataclass
class Move:
    """Representation of a Pokémon move."""
    id: str
    name: str
    move_type: PokemonType
    category: MoveCategory
    base_power: int = 0
    accuracy: float = 1.0  # 1.0 = 100%
    priority: int = 0
    pp: int = 16
    max_pp: int = 16
    is_contact: bool = False
    is_protect: bool = False
    is_heal: bool = False
    is_slicing: bool = False
    is_sound: bool = False
    is_reflectable: bool = False
    drain: Optional[Tuple[int, int]] = None
    recoil: Optional[Tuple[int, int]] = None
    boosts: Optional[Dict[str, int]] = None
    self_boosts: Optional[Dict[str, int]] = None
    target: str = "normal"
    always_hits: bool = False
    crit_ratio: int = 1
    will_crit: bool = False
    blocked_by_protect: bool = True
    secondaries: List[Dict] = field(default_factory=list)
    defrost: bool = False
    sleep_usable: bool = False
    sleep_talk_callable: bool = True
    bypass_substitute: bool = False
    volatile_status: Optional[str] = None

    def clone(self) -> "Move":
        """Copy mutable battle data without reloading or reinterpreting the Dex."""
        result = object.__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        result.secondaries = deepcopy(self.secondaries) if self.secondaries else []
        result.boosts = dict(self.boosts) if self.boosts is not None else None
        result.self_boosts = dict(self.self_boosts) if self.self_boosts is not None else None
        return result

    @classmethod
    def create(
        cls,
        name: str,
        move_type: PokemonType,
        category: MoveCategory,
        base_power: int = 0,
        accuracy: float = 1.0,
        priority: int = 0,
        pp: Optional[int] = None,
        max_pp: Optional[int] = None,
        is_contact: Optional[bool] = None,
        is_protect: Optional[bool] = None,
        is_heal: Optional[bool] = None,
        is_slicing: Optional[bool] = None,
        is_sound: Optional[bool] = None,
        is_reflectable: Optional[bool] = None,
        drain: Optional[Tuple[int, int]] = None,
        recoil: Optional[Tuple[int, int]] = None,
        boosts: Optional[Dict[str, int]] = None,
        self_boosts: Optional[Dict[str, int]] = None,
        target: Optional[str] = None,
        always_hits: Optional[bool] = None,
        crit_ratio: Optional[int] = None,
        will_crit: Optional[bool] = None,
        blocked_by_protect: Optional[bool] = None,
    ) -> "Move":
        move_id = clean_key(name)
        secondary_meta = {}
        if any(value is None for value in (pp, is_contact, is_protect, target, always_hits,
                                          crit_ratio, will_crit, blocked_by_protect)):
            try:
                from glaubermon.data.showdown_dex import ShowdownDex
                dex_m = ShowdownDex.get_instance().get_move(move_id)
                secondary_meta = {k:deepcopy(getattr(dex_m,k)) for k in ("secondaries","defrost","sleep_usable","sleep_talk_callable","bypass_substitute","volatile_status")}
                if pp is None:
                    pp = dex_m.pp
                if max_pp is None:
                    max_pp = dex_m.max_pp
                if is_contact is None:
                    is_contact = dex_m.is_contact
                if is_protect is None:
                    is_protect = dex_m.is_protect
                if is_heal is None:
                    is_heal = dex_m.is_heal
                if is_slicing is None:
                    is_slicing = dex_m.is_slicing
                if is_sound is None:
                    is_sound = dex_m.is_sound
                if is_reflectable is None:
                    is_reflectable = dex_m.is_reflectable
                if drain is None:
                    drain = dex_m.drain
                if recoil is None:
                    recoil = dex_m.recoil
                if boosts is None:
                    boosts = dex_m.boosts
                if self_boosts is None:
                    self_boosts = dex_m.self_boosts
                if target is None:
                    target = dex_m.target
                if always_hits is None:
                    always_hits = dex_m.always_hits
                if crit_ratio is None:
                    crit_ratio = dex_m.crit_ratio
                if will_crit is None:
                    will_crit = dex_m.will_crit
                if blocked_by_protect is None:
                    blocked_by_protect = dex_m.blocked_by_protect
            except Exception:
                pass

        return cls(
            id=move_id,
            name=name,
            move_type=move_type,
            category=category,
            base_power=base_power,
            accuracy=accuracy,
            priority=priority,
            pp=16 if pp is None else pp,
            max_pp=16 if max_pp is None else max_pp,
            is_contact=False if is_contact is None else is_contact,
            is_protect=False if is_protect is None else is_protect,
            is_heal=False if is_heal is None else is_heal,
            is_slicing=False if is_slicing is None else is_slicing,
            is_sound=False if is_sound is None else is_sound,
            is_reflectable=False if is_reflectable is None else is_reflectable,
            drain=drain,
            recoil=recoil,
            boosts=boosts,
            self_boosts=self_boosts,
            target="normal" if target is None else target,
            always_hits=False if always_hits is None else always_hits,
            crit_ratio=1 if crit_ratio is None else crit_ratio,
            will_crit=False if will_crit is None else will_crit,
            blocked_by_protect=True if blocked_by_protect is None else blocked_by_protect,
            **secondary_meta,
        )

    @classmethod
    def from_dex(cls, name_or_id: str) -> "Move":
        """Instantiate canonical move directly from ShowdownDex."""
        from glaubermon.data.showdown_dex import ShowdownDex
        return ShowdownDex.get_instance().get_move(name_or_id)


@dataclass
class Pokemon:
    """Representation of a Pokémon in battle."""
    species: str
    level: int = 100
    types: Tuple[PokemonType, Optional[PokemonType]] = (PokemonType.NORMAL, None)
    max_hp: int = 0
    current_hp: int = -1
    status: StatusCondition = StatusCondition.NONE
    status_turns: int = 0
    item: Optional[str] = None
    ability: Optional[str] = None
    tera_type: Optional[PokemonType] = None
    is_terastallized: bool = False
    moves: List[Move] = field(default_factory=list)
    boosts: Dict[str, int] = field(default_factory=lambda: {
        "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0
    })
    protect_streak: int = 0
    booster_stat: Optional[str] = None
    choice_locked_move: Optional[str] = None
    raw_stats: Dict[str, int] = field(default_factory=dict)
    volatiles: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 1. Fill species types if uninitialized and species exists in Dex
        if self.types == (PokemonType.NORMAL, None) and clean_key(self.species) not in (
            "normal", "eevee", "snorlax", "blissey", "chansey", "ditto", "porygon", "tauros"
        ):
            try:
                from glaubermon.data.showdown_dex import ShowdownDex
                t, _ = ShowdownDex.get_instance().get_pokemon_info(self.species)
                if t != (PokemonType.NORMAL, None):
                    self.types = t
            except Exception:
                pass

        # 2. Compute authentic cartridge stats if not explicitly provided
        if self.max_hp <= 0 or not self.raw_stats:
            try:
                from glaubermon.data.showdown_dex import ShowdownDex
                calc = ShowdownDex.get_instance().calculate_pokemon_stats(self.species, level=self.level)
                if self.max_hp <= 0:
                    self.max_hp = calc["hp"]
                if not self.raw_stats:
                    self.raw_stats = dict(calc)
            except Exception:
                if self.max_hp <= 0:
                    self.max_hp = 300
                if not self.raw_stats:
                    self.raw_stats = {"hp": 300, "atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}

        if self.current_hp == -1:
            self.current_hp = self.max_hp

        # 3. Synchronize form/item-dependent move types (e.g. Ivy Cudgel for Ogerpon forms)
        spec_clean = clean_key(self.species)
        it_clean = clean_key(self.item)
        for idx, m in enumerate(self.moves):
            m_clean = clean_key(getattr(m, "id", ""))
            if m_clean == "ivycudgel":
                if "wellspring" in spec_clean or "wellspring" in it_clean:
                    self.moves[idx].move_type = PokemonType.WATER
                elif "hearthflame" in spec_clean or "hearthflame" in it_clean:
                    self.moves[idx].move_type = PokemonType.FIRE
                elif "cornerstone" in spec_clean or "cornerstone" in it_clean:
                    self.moves[idx].move_type = PokemonType.ROCK
                elif spec_clean == "ogerpon" or "teal" in it_clean:
                    self.moves[idx].move_type = PokemonType.GRASS

    @property
    def is_fainted(self) -> bool:
        return self.current_hp <= 0

    @property
    def hp_percent(self) -> float:
        if self.max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, self.current_hp / self.max_hp))

    @property
    def active_types(self) -> Tuple[PokemonType, Optional[PokemonType]]:
        if self.is_terastallized and self.tera_type:
            return (self.tera_type, None)
        return self.types

    def get_booster_boosted_stat(self) -> Optional[str]:
        """Determine which stat is heightened by Protosynthesis or Quark Drive."""
        ability_clean = clean_key(self.ability)
        item_clean = clean_key(self.item)
        if "protosynthesis" not in ability_clean and "quarkdrive" not in ability_clean:
            return None
        if item_clean != "boosterenergy":
            return None

        candidates = ["atk", "def", "spa", "spd", "spe"]
        best_stat = "atk"
        best_val = -1
        for s in candidates:
            val = self.raw_stats.get(s, 100)
            if val > best_val:
                best_val = val
                best_stat = s
        return best_stat

    def effective_stat(self, stat_name: str, ignore_boosts: bool = False) -> int:
        """Compute the current in-battle stat value including stage boosts, items, and abilities."""
        base = self.raw_stats.get(stat_name, 100)
        stage = 0 if ignore_boosts else self.boosts.get(stat_name, 0)
        mult = STAT_STAGE_MULTIPLIERS.get(stage, 1.0)
        stat = int(base * mult)

        # Protosynthesis / Quark Drive booster calculation
        boosted = self.booster_stat or self.get_booster_boosted_stat()
        if boosted == stat_name:
            if stat_name == "spe":
                stat = int(stat * 1.5)
            else:
                stat = int(stat * 1.3)

        item_clean = clean_key(self.item)
        if stat_name == "spe":
            if self.status == StatusCondition.PARALYSIS:
                stat = int(stat * 0.5)
            if item_clean == "choicescarf":
                stat = int(stat * 1.5)
        elif stat_name == "atk":
            if item_clean == "choiceband":
                stat = int(stat * 1.5)
        elif stat_name == "spa":
            if item_clean == "choicespecs":
                stat = int(stat * 1.5)
        elif stat_name == "def":
            if item_clean == "eviolite":
                stat = int(stat * 1.5)
        elif stat_name == "spd":
            if item_clean in ("assaultvest", "eviolite"):
                stat = int(stat * 1.5)

        return max(1, stat)

    def take_damage(self, amount: int) -> int:
        """Apply damage and return actual damage taken."""
        actual = min(self.current_hp, max(0, amount))
        self.current_hp -= actual
        return actual

    def heal(self, amount: int) -> int:
        """Heal HP and return actual amount restored."""
        healed = min(self.max_hp - self.current_hp, max(0, amount))
        self.current_hp += healed
        return healed

    def is_grounded(self) -> bool:
        """Check if this Pokémon is affected by Ground moves, Spikes, and Toxic Spikes."""
        it = clean_key(self.item)
        if it == "airballoon":
            return False
        ab = clean_key(self.ability)
        if ab in ("levitate", "eartheater"):
            return False
        if PokemonType.FLYING in self.active_types:
            return False
        return True

    def calculate_hazard_damage(self, hazards: Dict[Any, int]) -> int:
        """Calculate exact entry hazard damage using active types, items, and abilities."""
        if not hazards:
            return 0
        it = clean_key(self.item)
        if it == "heavydutyboots":
            return 0
        ab = clean_key(self.ability)
        if ab == "magicguard":
            return 0

        total_dmg = 0
        # Stealth Rock (uses active_types to accurately account for Terastallization)
        if Hazard.STEALTH_ROCK in hazards:
            t1, t2 = self.active_types
            eff = get_type_effectiveness(PokemonType.ROCK, t1, t2)
            fraction = 0.125 * eff
            total_dmg += max(1,int(self.max_hp * fraction)) if eff else 0

        # Spikes (grounded check: Flying type, Levitate, Air Balloon)
        if self.is_grounded():
            spikes_lvl = hazards.get(Hazard.SPIKES_1, 0)
            if spikes_lvl == 1:
                total_dmg += max(1,self.max_hp // 8)
            elif spikes_lvl == 2:
                total_dmg += max(1,self.max_hp // 6)
            elif spikes_lvl >= 3:
                total_dmg += max(1,self.max_hp // 4)

        return total_dmg

    def is_dead_to_hazards(self, hazards: Dict[Any, int]) -> bool:
        """Return True if this Pokémon would immediately faint upon switching in."""
        if self.is_fainted:
            return True
        dmg = self.calculate_hazard_damage(hazards)
        return self.current_hp <= dmg

    def clone(self) -> "Pokemon":
        """Isolate branches, preserving runtime flags without running __post_init__."""
        result = object.__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        result.moves = [move.clone() for move in self.moves]
        result.boosts = dict(self.boosts)
        result.raw_stats = dict(self.raw_stats)
        result.volatiles = deepcopy(self.volatiles)
        return result
