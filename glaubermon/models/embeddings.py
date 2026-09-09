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

MOVE_DIM = 32
STAT_DIM = 64


def encode_move(move: Move) -> torch.Tensor:
    """Encode a single move into a 1D feature vector of size 32."""
    vec = torch.zeros(MOVE_DIM, dtype=torch.float32)
    vec[0] = move.base_power / 200.0
    vec[1] = move.accuracy
    vec[2] = (move.priority + 3) / 8.0  # Priority typically -3 to +5
    vec[3] = move.pp / 40.0

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

    return vec


def encode_pokemon(mon: Optional[Pokemon], is_active: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a single Pokémon and its 4 moves into moves (4, 32) + stats (64,)."""
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

    return moves_tensor, stats_tensor


def encode_battle_state(state: BattleState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert entire 6v6 BattleState into model-ready PyTorch tensors.

    Returns:
        p1_tensor: Tuple of (moves (6, 4, 32), stats (6, 64))
        p2_tensor: Tuple of (moves (6, 4, 32), stats (6, 64))
        field_tensor: Tensor of shape (16,) containing weather, terrain, hazards
    """
    p1_moves = []
    p1_stats = []
    for i in range(6):
        mon = state.p1.pokemon[i] if i < len(state.p1.pokemon) else None
        m_vec, s_vec = encode_pokemon(mon, is_active=(i == state.p1.active_index))
        p1_moves.append(m_vec)
        p1_stats.append(s_vec)

    p2_moves = []
    p2_stats = []
    for i in range(6):
        mon = state.p2.pokemon[i] if i < len(state.p2.pokemon) else None
        m_vec, s_vec = encode_pokemon(mon, is_active=(i == state.p2.active_index))
        p2_moves.append(m_vec)
        p2_stats.append(s_vec)

    p1_moves_t = torch.stack(p1_moves)  # (6, 4, 32)
    p1_stats_t = torch.stack(p1_stats)  # (6, 64)
    p2_moves_t = torch.stack(p2_moves)  # (6, 4, 32)
    p2_stats_t = torch.stack(p2_stats)  # (6, 64)

    field_tensor = torch.zeros(16, dtype=torch.float32)
    w_idx = WEATHER_MAP.get(state.weather, 0)
    field_tensor[w_idx % 8] = 1.0
    t_idx = TERRAIN_MAP.get(state.terrain, 0)
    field_tensor[8 + (t_idx % 5)] = 1.0
    if Hazard.STEALTH_ROCK in state.p1.hazards:
        field_tensor[13] = 1.0
    if Hazard.STEALTH_ROCK in state.p2.hazards:
        field_tensor[14] = 1.0
    field_tensor[15] = min(1.0, state.turn / 50.0)

    p1_combined = (p1_moves_t, p1_stats_t)
    p2_combined = (p2_moves_t, p2_stats_t)
    return p1_combined, p2_combined, field_tensor
