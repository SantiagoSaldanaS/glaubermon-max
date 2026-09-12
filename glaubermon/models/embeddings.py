"""Feature extraction and rich tensor encodings for Pokémon battle states."""

from typing import Dict, List, Optional, Tuple
import torch
from glaubermon.core.battle_state import BattleState
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, StatusCondition, Weather, Terrain, Hazard

TYPE_MAP = {t: i for i, t in enumerate(PokemonType)}
STATUS_MAP = {s: i for i, s in enumerate(StatusCondition)}
WEATHER_MAP = {w: i for i, w in enumerate(Weather)}
TERRAIN_MAP = {t: i for i, t in enumerate(Terrain)}

MOVE_DIM = 33
STAT_DIM = 83
FIELD_DIM = 40
FEATURE_SCHEMA = "public_history_v6"


_MOVE_ENCODING_CACHE: Dict[Tuple, torch.Tensor] = {}


def encode_move(move: Move) -> torch.Tensor:
    """Encode a single move into a 1D feature vector of size MOVE_DIM (with memoization)."""
    key = (move.base_power, move.accuracy, move.priority, move.pp, move.move_type, move.category, move.pp_known)
    cached = _MOVE_ENCODING_CACHE.get(key)
    if cached is not None:
        return cached.clone()

    vec = torch.zeros(MOVE_DIM, dtype=torch.float32)
    vec[0] = move.base_power / 200.0
    vec[1] = move.accuracy
    vec[2] = (move.priority + 3) / 8.0  # Priority typically -3 to +5
    vec[3] = move.pp / 40.0
    vec[32] = float(move.pp_known)

    # Move Type one-hot (indices 4 to 22)
    type_idx = TYPE_MAP.get(move.move_type, 0)
    vec[4 + (type_idx % 19)] = 1.0

    # Category one-hot (indices 23, 24, 25)
    if move.category == MoveCategory.PHYSICAL:
        vec[23] = 1.0
    elif move.category == MoveCategory.SPECIAL:
        vec[24] = 1.0
    elif move.category == MoveCategory.STATUS:
        vec[25] = 1.0

    if len(_MOVE_ENCODING_CACHE) >= 4096:
        _MOVE_ENCODING_CACHE.clear()
    _MOVE_ENCODING_CACHE[key] = vec
    return vec.clone()


def encode_volatiles(mon):
    sub = mon.volatiles.get("substitute",{}).get("hp",0)
    taunt = mon.volatiles.get("taunt",{}).get("duration",0)
    bind = mon.volatiles.get("partiallytrapped",{}).get("duration",0)
    if "trapped" in mon.volatiles or "request_trapped" in mon.volatiles: bind = -1
    confusion = mon.volatiles.get("confusion",{}).get("time",0)
    return torch.tensor([sub / max(1,mon.max_hp) if sub >= 0 else -1,
                         taunt / 4, bind / 8, confusion / 5],dtype=torch.float32)


def encode_restrictions(mon):
    # Presence is separate from move identity: public logs can leave it unknown.
    result = torch.zeros(15,dtype=torch.float32)
    for i,key in enumerate(("encore","disable","leechseed")):
        result[i] = float(key in mon.volatiles)
    for offset,move_id in ((3,mon.volatiles.get("encore",{}).get("move")),
                           (7,mon.volatiles.get("disable",{}).get("move")),(11,mon.last_move)):
        for slot,move in enumerate(mon.moves[:4]):
            result[offset+slot] = float(move.id == move_id)
    return result


def encode_pokemon(mon: Optional[Pokemon], is_active: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a single Pokémon and its 4 moves into moves (4, MOVE_DIM) + stats (STAT_DIM,)."""
    moves_tensor = torch.zeros(4, MOVE_DIM, dtype=torch.float32)
    stats_tensor = torch.zeros(STAT_DIM, dtype=torch.float32)

    if mon is None or mon.is_fainted:
        stats_tensor[0] = 1.0  # is_fainted flag
        return moves_tensor, stats_tensor

    # 1. Encode revealed moves
    for i, m in enumerate(mon.moves[:4]):
        moves_tensor[i] = encode_move(m)

    # 2. Status and state flags
    stats_tensor[1] = mon.hp_percent
    stats_tensor[2] = 1.0 if is_active else 0.0
    stats_tensor[3] = 1.0 if mon.is_terastallized else 0.0

    # 3. Primary & Secondary types
    types = mon.active_types
    t1_idx = TYPE_MAP.get(types[0], 0)
    stats_tensor[4 + (t1_idx % 19)] = 1.0
    if types[1] is not None:
        t2_idx = TYPE_MAP.get(types[1], 0)
        stats_tensor[23 + (t2_idx % 19)] = 1.0

    # 4. Status condition (42..48)
    s_idx = STATUS_MAP.get(mon.status, 0)
    stats_tensor[42 + (s_idx % 7)] = 1.0

    # 5. Stage boosts (atk, def, spa, spd, spe) normalized to [-1, 1] (49..53)
    boost_keys = ["atk", "def", "spa", "spd", "spe"]
    for i, k in enumerate(boost_keys):
        stats_tensor[49 + i] = mon.boosts.get(k, 0) / 6.0

    # 6. Normalized Base Stats (54..59)
    stat_keys = ["hp", "atk", "def", "spa", "spd", "spe"]
    for i, k in enumerate(stat_keys):
        val = mon.raw_stats.get(k, 80) if mon.raw_stats else 80
        stats_tensor[54 + i] = min(1.0, val / 255.0)

    stats_tensor[64:68] = encode_volatiles(mon)
    stats_tensor[68:83] = encode_restrictions(mon)
    return moves_tensor, stats_tensor


def encode_battle_state(state: BattleState) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Convert entire 6v6 BattleState into model-ready PyTorch tensors with preallocated buffers.

    Returns:
        p1_tensor: Tuple of (moves (6, 4, MOVE_DIM), stats (6, STAT_DIM))
        p2_tensor: Tuple of (moves (6, 4, MOVE_DIM), stats (6, STAT_DIM))
        field_tensor: Tensor of shape (40,) containing weather, terrain, hazards
    """
    p1_moves_t = torch.zeros(6, 4, MOVE_DIM, dtype=torch.float32)
    p1_stats_t = torch.zeros(6, STAT_DIM, dtype=torch.float32)
    p2_moves_t = torch.zeros(6, 4, MOVE_DIM, dtype=torch.float32)
    p2_stats_t = torch.zeros(6, STAT_DIM, dtype=torch.float32)

    for i in range(6):
        mon = state.p1.pokemon[i] if i < len(state.p1.pokemon) else None
        if mon and not mon.is_fainted:
            for j in range(min(4, len(mon.moves))):
                p1_moves_t[i, j] = encode_move(mon.moves[j])
            p1_stats_t[i, 64:68] = encode_volatiles(mon)
            p1_stats_t[i, 68:83] = encode_restrictions(mon)
            p1_stats_t[i, 1] = mon.hp_percent
            p1_stats_t[i, 2] = 1.0 if i == state.p1.active_index else 0.0
            p1_stats_t[i, 3] = 1.0 if mon.is_terastallized else 0.0
            t1_idx = TYPE_MAP.get(mon.active_types[0], 0)
            p1_stats_t[i, 4 + (t1_idx % 19)] = 1.0
            if mon.active_types[1] is not None:
                t2_idx = TYPE_MAP.get(mon.active_types[1], 0)
                p1_stats_t[i, 23 + (t2_idx % 19)] = 1.0
            s_idx = STATUS_MAP.get(mon.status, 0)
            p1_stats_t[i, 42 + (s_idx % 7)] = 1.0
            for k_i, k in enumerate(["atk", "def", "spa", "spd", "spe"]):
                p1_stats_t[i, 49 + k_i] = mon.boosts.get(k, 0) / 6.0
            if mon.raw_stats:
                for k_i, k in enumerate(["hp", "atk", "def", "spa", "spd", "spe"]):
                    val = mon.raw_stats.get(k, 80)
                    p1_stats_t[i, 54 + k_i] = min(1.0, val / 255.0)
            else:
                p1_stats_t[i, 54:60] = 80.0 / 255.0
        else:
            p1_stats_t[i, 0] = 1.0

    for i in range(6):
        mon = state.p2.pokemon[i] if i < len(state.p2.pokemon) else None
        if mon and not mon.is_fainted:
            for j in range(min(4, len(mon.moves))):
                p2_moves_t[i, j] = encode_move(mon.moves[j])
            p2_stats_t[i, 64:68] = encode_volatiles(mon)
            p2_stats_t[i, 68:83] = encode_restrictions(mon)
            p2_stats_t[i, 1] = mon.hp_percent
            p2_stats_t[i, 2] = 1.0 if i == state.p2.active_index else 0.0
            p2_stats_t[i, 3] = 1.0 if mon.is_terastallized else 0.0
            t1_idx = TYPE_MAP.get(mon.active_types[0], 0)
            p2_stats_t[i, 4 + (t1_idx % 19)] = 1.0
            if mon.active_types[1] is not None:
                t2_idx = TYPE_MAP.get(mon.active_types[1], 0)
                p2_stats_t[i, 23 + (t2_idx % 19)] = 1.0
            s_idx = STATUS_MAP.get(mon.status, 0)
            p2_stats_t[i, 42 + (s_idx % 7)] = 1.0
            for k_i, k in enumerate(["atk", "def", "spa", "spd", "spe"]):
                p2_stats_t[i, 49 + k_i] = mon.boosts.get(k, 0) / 6.0
            if mon.raw_stats:
                for k_i, k in enumerate(["hp", "atk", "def", "spa", "spd", "spe"]):
                    val = mon.raw_stats.get(k, 80)
                    p2_stats_t[i, 54 + k_i] = min(1.0, val / 255.0)
            else:
                p2_stats_t[i, 54:60] = 80.0 / 255.0
        else:
            p2_stats_t[i, 0] = 1.0

    field_tensor = torch.zeros(FIELD_DIM, dtype=torch.float32)
    w_idx = WEATHER_MAP.get(state.weather, 0)
    field_tensor[w_idx % 8] = 1.0
    t_idx = TERRAIN_MAP.get(state.terrain, 0)
    field_tensor[8 + (t_idx % 5)] = 1.0
    if Hazard.STEALTH_ROCK in state.p1.hazards:
        field_tensor[13] = 1.0
    if Hazard.STEALTH_ROCK in state.p2.hazards:
        field_tensor[14] = 1.0
    field_tensor[15] = min(1.0, state.turn / 50.0)

    # Additional public features; negative duration means active, duration unknown.
    field_tensor[16] = state.trick_room / 5.0
    field_tensor[17] = state.weather_turns / 8.0
    field_tensor[18] = state.terrain_turns / 8.0
    field_tensor[37] = float(1 in state.pending_switches)
    field_tensor[38] = float(2 in state.pending_switches)
    field_tensor[39] = float(state.continuation is not None)
    for offset, side in ((19,state.p1),(28,state.p2)):
        for i, name in enumerate(("reflect","lightscreen","auroraveil")):
            field_tensor[offset+i] = side.screens.get(name,0) / 8.0
        field_tensor[offset+3] = side.tailwind / 4.0
        field_tensor[offset+4] = side.hazards.get(Hazard.SPIKES_1,0) / 3.0
        field_tensor[offset+5] = side.hazards.get(Hazard.TOXIC_SPIKES_1,0) / 2.0
        field_tensor[offset+6] = float(bool(side.hazards.get(Hazard.STICKY_WEB,0)))

    p1_combined = (p1_moves_t, p1_stats_t)
    p2_combined = (p2_moves_t, p2_stats_t)
    return p1_combined, p2_combined, field_tensor
