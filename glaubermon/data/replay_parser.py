"""Parser to convert Showdown replay logs into supervised training datasets."""

import os
import re
import json
import glob
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from tqdm import tqdm

from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, Hazard, Weather, Terrain
from glaubermon.models.embeddings import encode_battle_state


def clean_name(name: str) -> str:
    """Clean species and move names."""
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()


class ReplayParser:
    """Parses raw Showdown replay JSONs into (state_tensors, action_idx, outcome) datasets."""

    def __init__(self, replays_dir: str = "data/replays"):
        self.replays_dir = replays_dir

    def parse_all(self, max_replays: Optional[int] = None) -> List[Tuple]:
        files = glob.glob(os.path.join(self.replays_dir, "*.json"))
        if max_replays:
            files = files[:max_replays]

        print(f"Parsing {len(files)} replay files into training samples...")
        dataset = []

        for fpath in tqdm(files, desc="Parsing Replays"):
            try:
                samples = self.parse_single_replay(fpath)
                dataset.extend(samples)
            except Exception:
                continue

        print(f"Successfully extracted {len(dataset)} expert turn samples!")
        return dataset

    def parse_single_replay(self, filepath: str) -> List[Tuple]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        log = data.get("log", "")
        lines = log.split("\n")

        p1_name = None
        p2_name = None
        p1_team_names = []
        p2_team_names = []
        winner = None

        # Phase 1: Header & Teams
        for line in lines:
            parts = line.split("|")
            if len(parts) < 2:
                continue
            cmd = parts[1]

            if cmd == "player":
                if len(parts) >= 4:
                    if parts[2] == "p1":
                        p1_name = parts[3]
                    elif parts[2] == "p2":
                        p2_name = parts[3]
            elif cmd == "poke":
                if len(parts) >= 4:
                    species = parts[3].split(",")[0].strip()
                    if parts[2] == "p1":
                        p1_team_names.append(species)
                    elif parts[2] == "p2":
                        p2_team_names.append(species)
            elif cmd == "win":
                if len(parts) >= 3:
                    winner = parts[2].strip()

        if not p1_team_names or not p2_team_names or not winner:
            return []

        p1_outcome = 1.0 if winner == p1_name else -1.0
        p2_outcome = -p1_outcome

        # Helper to construct initial Pokemon objects
        def build_initial_team(species_list):
            mons = []
            for sp in species_list[:6]:
                mons.append(Pokemon(
                    species=sp,
                    types=(PokemonType.NORMAL, None),
                    max_hp=300, current_hp=300,
                    moves=[],
                    raw_stats={"hp": 300, "atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}
                ))
            return mons

        p1_mons = build_initial_team(p1_team_names)
        p2_mons = build_initial_team(p2_team_names)
        state = BattleState(
            p1=BattleSide(pokemon=p1_mons),
            p2=BattleSide(pokemon=p2_mons)
        )

        samples = []
        current_turn_p1_action: Optional[int] = None
        current_turn_p2_action: Optional[int] = None

        # Phase 2: Iterate battle lines
        for line in lines:
            parts = line.split("|")
            if len(parts) < 2:
                continue
            cmd = parts[1]

            if cmd == "turn":
                # Save previous turn samples if recorded
                if current_turn_p1_action is not None:
                    tensors = encode_battle_state(state)
                    samples.append((tensors, current_turn_p1_action, p1_outcome))
                if current_turn_p2_action is not None:
                    # Invert state perspective for P2
                    inv_state = BattleState(p1=state.p2, p2=state.p1, weather=state.weather, terrain=state.terrain, turn=state.turn)
                    tensors = encode_battle_state(inv_state)
                    samples.append((tensors, current_turn_p2_action, p2_outcome))

                current_turn_p1_action = None
                current_turn_p2_action = None
                try:
                    state.turn = int(parts[2])
                except Exception:
                    state.turn += 1

            elif cmd == "switch":
                # |switch|p1a: Dragapult|Dragapult, M|100/100
                if len(parts) >= 4:
                    ident = parts[2]
                    sp = parts[3].split(",")[0].strip()
                    is_p1 = "p1" in ident
                    side = state.p1 if is_p1 else state.p2

                    # Find slot index in team
                    slot = -1
                    for idx, mon in enumerate(side.pokemon):
                        if clean_name(mon.species) == clean_name(sp):
                            slot = idx
                            break
                    if slot != -1:
                        side.active_index = slot
                        action_idx = 8 + min(slot, 4)  # Switch action mapped to 8..12
                        if is_p1 and current_turn_p1_action is None:
                            current_turn_p1_action = action_idx
                        elif not is_p1 and current_turn_p2_action is None:
                            current_turn_p2_action = action_idx

            elif cmd == "move":
                # |move|p1a: Dragapult|Draco Meteor|p2a: Ting-Lu
                if len(parts) >= 4:
                    ident = parts[2]
                    move_name = parts[3].strip()
                    is_p1 = "p1" in ident
                    side = state.p1 if is_p1 else state.p2
                    active = side.active_pokemon

                    if active:
                        # Find or add move to active mon's revealed moves
                        move_id = clean_name(move_name)
                        move_idx = -1
                        for idx, m in enumerate(active.moves):
                            if m.id == move_id:
                                move_idx = idx
                                break
                        if move_idx == -1 and len(active.moves) < 4:
                            new_m = Move.create(move_name, PokemonType.NORMAL, MoveCategory.PHYSICAL, base_power=80)
                            active.moves.append(new_m)
                            move_idx = len(active.moves) - 1

                        if move_idx != -1:
                            is_tera = "[terastallize]" in line or active.is_terastallized
                            action_idx = (move_idx + 4) if is_tera else move_idx
                            if is_p1 and current_turn_p1_action is None:
                                current_turn_p1_action = action_idx
                            elif not is_p1 and current_turn_p2_action is None:
                                current_turn_p2_action = action_idx

            elif cmd == "-damage":
                # |-damage|p1a: Dragapult|50/100
                if len(parts) >= 4:
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

            elif cmd == "-terastallize":
                # |-terastallize|p1a: Kingambit|Flying
                if len(parts) >= 4:
                    side = state.p1 if "p1" in parts[2] else state.p2
                    if side.active_pokemon:
                        side.active_pokemon.is_terastallized = True
                        side.is_tera_used = True

        return samples


if __name__ == "__main__":
    parser = ReplayParser()
    dataset = parser.parse_all(max_replays=10)
    print(f"Sample dataset size from 10 replays: {len(dataset)}")
