"""Mass-scale Parquet Replay Dataset Loader with True Showdown Pokédex Lookups."""

import os
import requests
import pandas as pd
from typing import List, Tuple, Optional
from tqdm import tqdm

from glaubermon.data.replay_parser import clean_name
from glaubermon.data.showdown_dex import ShowdownDex
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory
from glaubermon.models.embeddings import encode_battle_state


def parse_raw_log(log_str: str) -> List[Tuple]:
    """Parse raw Showdown battle log string into training samples with real Dex stats."""
    dex = ShowdownDex.get_instance()
    lines = log_str.split("\n")
    p1_name = None
    p2_name = None
    p1_team_names = []
    p2_team_names = []
    winner = None

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]
        if cmd == "player" and len(parts) >= 4:
            if parts[2] == "p1":
                p1_name = parts[3]
            elif parts[2] == "p2":
                p2_name = parts[3]
        elif cmd == "poke" and len(parts) >= 4:
            species = parts[3].split(",")[0].strip()
            if parts[2] == "p1":
                p1_team_names.append(species)
            elif parts[2] == "p2":
                p2_team_names.append(species)
        elif cmd == "win" and len(parts) >= 3:
            winner = parts[2].strip()

    if not p1_team_names or not p2_team_names or not winner:
        return []

    p1_outcome = 1.0 if winner == p1_name else -1.0
    p2_outcome = -p1_outcome

    def build_initial_team(species_list):
        mons = []
        for sp in species_list[:6]:
            types, base_stats = dex.get_pokemon_info(sp)
            max_hp = base_stats.get("hp", 80) * 2 + 141  # Standard competitive Lv 100 HP
            mons.append(Pokemon(
                species=sp,
                types=types,
                max_hp=max_hp, current_hp=max_hp,
                moves=[],
                raw_stats=base_stats
            ))
        return mons

    p1_mons = build_initial_team(p1_team_names)
    p2_mons = build_initial_team(p2_team_names)
    state = BattleState(p1=BattleSide(pokemon=p1_mons), p2=BattleSide(pokemon=p2_mons))

    samples = []
    current_turn_p1_action = None
    current_turn_p2_action = None

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]

        if cmd == "turn":
            if current_turn_p1_action is not None:
                tensors = encode_battle_state(state)
                samples.append((tensors, current_turn_p1_action, p1_outcome))
            if current_turn_p2_action is not None:
                inv_state = BattleState(p1=state.p2, p2=state.p1, weather=state.weather, terrain=state.terrain, turn=state.turn)
                tensors = encode_battle_state(inv_state)
                samples.append((tensors, current_turn_p2_action, p2_outcome))

            current_turn_p1_action = None
            current_turn_p2_action = None
            try:
                state.turn = int(parts[2])
            except Exception:
                state.turn += 1

        elif cmd == "switch" and len(parts) >= 4:
            ident = parts[2]
            sp = parts[3].split(",")[0].strip()
            is_p1 = "p1" in ident
            side = state.p1 if is_p1 else state.p2

            slot = -1
            for idx, mon in enumerate(side.pokemon):
                if clean_name(mon.species) == clean_name(sp):
                    slot = idx
                    break
            if slot != -1:
                side.active_index = slot
                action_idx = 8 + min(slot, 4)
                if is_p1 and current_turn_p1_action is None:
                    current_turn_p1_action = action_idx
                elif not is_p1 and current_turn_p2_action is None:
                    current_turn_p2_action = action_idx

        elif cmd == "move" and len(parts) >= 4:
            ident = parts[2]
            move_name = parts[3].strip()
            is_p1 = "p1" in ident
            side = state.p1 if is_p1 else state.p2
            active = side.active_pokemon

            if active:
                move_id = clean_name(move_name)
                move_idx = -1
                for idx, m in enumerate(active.moves):
                    if m.id == move_id:
                        move_idx = idx
                        break
                if move_idx == -1 and len(active.moves) < 4:
                    m_type, m_cat, bp, acc, prio = dex.get_move_info(move_name)
                    new_m = Move.create(move_name, m_type, m_cat, base_power=bp, accuracy=acc / 100.0, priority=prio)
                    active.moves.append(new_m)
                    move_idx = len(active.moves) - 1

                if move_idx != -1:
                    is_tera = "[terastallize]" in line or active.is_terastallized
                    action_idx = (move_idx + 4) if is_tera else move_idx
                    if is_p1 and current_turn_p1_action is None:
                        current_turn_p1_action = action_idx
                    elif not is_p1 and current_turn_p2_action is None:
                        current_turn_p2_action = action_idx

        elif cmd == "-damage" and len(parts) >= 4:
            ident = parts[2]
            hp_str = parts[3].split()[0]
            side = state.p1 if "p1" in ident else state.p2
            active = side.active_pokemon
            if active and "/" in hp_str:
                cur, max_ = hp_str.split("/")
                try:
                    active.current_hp = int(float(cur) / float(max_) * active.max_hp)
                except Exception:
                    pass
            elif active and hp_str == "0":
                active.current_hp = 0

        elif cmd == "-terastallize" and len(parts) >= 4:
            side = state.p1 if "p1" in parts[2] else state.p2
            if side.active_pokemon:
                side.active_pokemon.is_terastallized = True
                side.is_tera_used = True

    return samples


def download_parquet_part(part_num: int, cache_dir: str = "data") -> str:
    """Download parquet file if not already cached locally."""
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"part-{part_num:05d}.parquet"
    local_path = os.path.join(cache_dir, filename)
    if os.path.exists(local_path):
        return local_path

    url = f"https://huggingface.co/datasets/milkkarten/pokemon-showdown-replays-merged/resolve/main/data/{filename}"
    print(f"Downloading {filename} (68 MB) from Hugging Face...")
    resp = requests.get(url, stream=True)
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {filename} to {local_path}.")
    return local_path


def load_mass_dataset(
    part_numbers: List[int] = [50],
    min_rating: int = 1500,
    max_games: Optional[int] = None,
    cache_path: str = "data/grandmaster_dataset_cache.pt",
    use_cache: bool = True,
    workers: int = 24
) -> List[Tuple]:
    """Download, parse in parallel on 32 CPU cores, and cache high-Elo Gen 9 OU games."""
    import torch
    from concurrent.futures import ProcessPoolExecutor

    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached grandmaster dataset from {cache_path}...")
        try:
            return torch.load(cache_path)
        except Exception:
            print("Cache corrupted or incompatible, rebuilding...")

    all_logs = []
    for p in part_numbers:
        path = download_parquet_part(p)
        print(f"Reading {path}...")
        df = pd.read_parquet(path, columns=["id", "rating", "log"])
        filtered = df[df["rating"] >= min_rating]
        print(f"Found {len(filtered)} games with rating >= {min_rating} in part {p}.")
        all_logs.extend(filtered["log"].tolist())

    if max_games:
        all_logs = all_logs[:max_games]

    print(f"\nParsing {len(all_logs)} elite games in parallel across {workers} CPU workers with ShowdownDex...")
    dataset = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(pool.map(parse_raw_log, all_logs, chunksize=25), total=len(all_logs), desc="Extracting Grandmaster States"))

    for r in results:
        dataset.extend(r)

    print(f"Mass dataset complete: {len(dataset):,} grandmaster decision states extracted!")

    # Save to disk cache for instantaneous future loads
    try:
        print(f"Saving dataset cache to {cache_path}...")
        torch.save(dataset, cache_path)
        print("Cache saved successfully.")
    except Exception as e:
        print(f"Failed to cache dataset: {e}")

    return dataset


if __name__ == "__main__":
    ds = load_mass_dataset(part_numbers=[50], min_rating=1600, max_games=50)
    print("Sample size:", len(ds))
