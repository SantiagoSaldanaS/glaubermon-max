"""Self-Play Policy & Value Training Loop for Glaubermon Max on RTX 4090."""

import os
import random
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, ActionType, Hazard
from glaubermon.models.embeddings import encode_battle_state
from glaubermon.models.set_transformer import GlaubermonMaxNet
from glaubermon.search.subgame_resolver import SubgameResolver, simulate_turn_transition
from glaubermon.search.evaluators import StateEvaluator


class NeuralEvaluator(StateEvaluator):
    """Evaluator that uses GlaubermonMaxNet to evaluate leaf positions."""

    def __init__(self, model: GlaubermonMaxNet, device: torch.device):
        self.model = model
        self.device = device

    def evaluate(self, state: BattleState) -> float:
        if state.is_game_over:
            return 1.0 if state.winner == 1 else -1.0

        (p1_m, p1_s), (p2_m, p2_s), field = encode_battle_state(state)
        p1_m = p1_m.to(self.device)
        p1_s = p1_s.to(self.device)
        p2_m = p2_m.to(self.device)
        p2_s = p2_s.to(self.device)
        field = field.to(self.device)

        with torch.no_grad():
            val, _ = self.model(p1_m, p1_s, p2_m, p2_s, field)
        return float(val.item())


def generate_random_battle() -> BattleState:
    """Generate diverse competitive 3v3 / 6v6 positions for training."""
    pool = [
        ("Great Tusk", (PokemonType.GROUND, PokemonType.FIGHTING), 371, 361, 298, 127, 142, 273, [
            Move.create("Close Combat", PokemonType.FIGHTING, MoveCategory.PHYSICAL, 120),
            Move.create("Headlong Rush", PokemonType.GROUND, MoveCategory.PHYSICAL, 120),
            Move.create("Ice Spinner", PokemonType.ICE, MoveCategory.PHYSICAL, 80),
            Move.create("Rapid Spin", PokemonType.NORMAL, MoveCategory.PHYSICAL, 50)
        ]),
        ("Dragapult", (PokemonType.DRAGON, PokemonType.GHOST), 317, 257, 186, 299, 186, 421, [
            Move.create("Shadow Ball", PokemonType.GHOST, MoveCategory.SPECIAL, 80),
            Move.create("Draco Meteor", PokemonType.DRAGON, MoveCategory.SPECIAL, 130),
            Move.create("Flamethrower", PokemonType.FIRE, MoveCategory.SPECIAL, 90),
            Move.create("U-turn", PokemonType.BUG, MoveCategory.PHYSICAL, 70)
        ]),
        ("Kingambit", (PokemonType.DARK, PokemonType.STEEL), 404, 405, 276, 140, 206, 136, [
            Move.create("Kowtow Cleave", PokemonType.DARK, MoveCategory.PHYSICAL, 85),
            Move.create("Sucker Punch", PokemonType.DARK, MoveCategory.PHYSICAL, 70, priority=1),
            Move.create("Iron Head", PokemonType.STEEL, MoveCategory.PHYSICAL, 80),
            Move.create("Swords Dance", PokemonType.NORMAL, MoveCategory.STATUS, 0)
        ]),
        ("Gholdengo", (PokemonType.STEEL, PokemonType.GHOST), 315, 140, 226, 399, 218, 267, [
            Move.create("Make It Rain", PokemonType.STEEL, MoveCategory.SPECIAL, 120),
            Move.create("Shadow Ball", PokemonType.GHOST, MoveCategory.SPECIAL, 80),
            Move.create("Focus Blast", PokemonType.FIGHTING, MoveCategory.SPECIAL, 120),
            Move.create("Nasty Plot", PokemonType.DARK, MoveCategory.STATUS, 0)
        ]),
        ("Dondozo", (PokemonType.WATER, None), 504, 236, 361, 149, 166, 106, [
            Move.create("Liquidation", PokemonType.WATER, MoveCategory.PHYSICAL, 85),
            Move.create("Body Press", PokemonType.FIGHTING, MoveCategory.PHYSICAL, 80),
            Move.create("Rest", PokemonType.PSYCHIC, MoveCategory.STATUS, 0),
            Move.create("Sleep Talk", PokemonType.NORMAL, MoveCategory.STATUS, 0)
        ]),
        ("Iron Valiant", (PokemonType.FAIRY, PokemonType.FIGHTING), 289, 296, 216, 339, 156, 364, [
            Move.create("Moonblast", PokemonType.FAIRY, MoveCategory.SPECIAL, 95),
            Move.create("Close Combat", PokemonType.FIGHTING, MoveCategory.PHYSICAL, 120),
            Move.create("Knock Off", PokemonType.DARK, MoveCategory.PHYSICAL, 65),
            Move.create("Thunderbolt", PokemonType.ELECTRIC, MoveCategory.SPECIAL, 90)
        ])
    ]

    p1_choices = random.sample(pool, 3)
    p2_choices = random.sample(pool, 3)

    def build_team(choices):
        mons = []
        for name, types, hp, atk, def_, spa, spd, spe, moves in choices:
            mons.append(Pokemon(
                species=name,
                types=types,
                max_hp=hp,
                current_hp=hp,
                moves=[m for m in moves],
                raw_stats={"hp": hp, "atk": atk, "def": def_, "spa": spa, "spd": spd, "spe": spe},
                tera_type=types[0]
            ))
        return mons

    return BattleState(
        p1=BattleSide(pokemon=build_team(p1_choices), hazards={Hazard.STEALTH_ROCK: 1}),
        p2=BattleSide(pokemon=build_team(p2_choices), hazards={Hazard.STEALTH_ROCK: 1})
    )


def run_self_play_game(resolver: SubgameResolver) -> List[Tuple]:
    """Simulate a full self-play match between two resolvers, collecting training samples."""
    state = generate_random_battle()
    trajectory = []

    for turn in range(30):
        if state.is_game_over:
            break

        # Extract state tensors
        state_tensors = encode_battle_state(state)

        # Solve turn for P1
        act1, p1_strat, actions1, _ = resolver.resolve_turn(state, depth=1, sample=True)

        # Invert state perspective to solve for P2
        inv_state = BattleState(p1=state.p2, p2=state.p1, weather=state.weather, terrain=state.terrain, turn=state.turn)
        act2, _, _, _ = resolver.resolve_turn(inv_state, depth=1, sample=True)

        # Record P1 transition
        trajectory.append((state_tensors, p1_strat, len(actions1)))

        # Advance state
        state = simulate_turn_transition(state, act1, act2)

        # Auto-switch fainted Pokémon
        if state.p1.active_pokemon.is_fainted and not state.p1.is_all_fainted:
            sw = state.p1.available_switches()
            if sw:
                state.p1.active_index = sw[0]
        if state.p2.active_pokemon.is_fainted and not state.p2.is_all_fainted:
            sw = state.p2.available_switches()
            if sw:
                state.p2.active_index = sw[0]

    winner = state.winner or 0
    outcome = 1.0 if winner == 1 else (-1.0 if winner == 2 else 0.0)

    # Attach outcome Z to each step
    dataset = []
    for state_t, target_policy, num_acts in trajectory:
        dataset.append((state_t, target_policy, num_acts, outcome))

    return dataset


def train_glaubermon(num_games: int = 20, batch_size: int = 16, epochs: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Glaubermon Max on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = GlaubermonMaxNet(d_model=128, nhead=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    evaluator = NeuralEvaluator(model, device)
    resolver = SubgameResolver(evaluator=evaluator)

    print(f"\n--- Phase 1: Generating {num_games} Self-Play Trajectories ---")
    all_data = []
    for g in tqdm(range(num_games), desc="Self-Play Games"):
        game_samples = run_self_play_game(resolver)
        all_data.extend(game_samples)

    print(f"\nCollected {len(all_data)} decision states. Training for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        random.shuffle(all_data)
        epoch_v_loss = 0.0
        epoch_p_loss = 0.0
        steps = 0

        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i + batch_size]
            if not batch:
                continue

            p1_m_list, p1_s_list, p2_m_list, p2_s_list, f_list = [], [], [], [], []
            val_targets = []
            policy_targets = []

            for state_tensors, pol_target, num_acts, outcome in batch:
                (p1_m, p1_s), (p2_m, p2_s), field = state_tensors
                p1_m_list.append(p1_m)
                p1_s_list.append(p1_s)
                p2_m_list.append(p2_m)
                p2_s_list.append(p2_s)
                f_list.append(field)
                val_targets.append(outcome)

                # Target action distribution padded to 14
                target_dist = np.zeros(14, dtype=np.float32)
                target_dist[:min(num_acts, 14)] = pol_target[:min(num_acts, 14)]
                policy_targets.append(torch.tensor(target_dist))

            p1_m_b = torch.stack(p1_m_list).to(device)
            p1_s_b = torch.stack(p1_s_list).to(device)
            p2_m_b = torch.stack(p2_m_list).to(device)
            p2_s_b = torch.stack(p2_s_list).to(device)
            f_b = torch.stack(f_list).to(device)
            v_targets_b = torch.tensor(val_targets, dtype=torch.float32, device=device).unsqueeze(-1)
            p_targets_b = torch.stack(policy_targets).to(device)

            optimizer.zero_grad()
            pred_v, pred_p = model(p1_m_b, p1_s_b, p2_m_b, p2_s_b, f_b)

            loss_v = value_loss_fn(pred_v, v_targets_b)
            loss_p = policy_loss_fn(pred_p, p_targets_b)
            loss = loss_v + loss_p

            loss.backward()
            optimizer.step()

            epoch_v_loss += loss_v.item()
            epoch_p_loss += loss_p.item()
            steps += 1

        print(f"Epoch {epoch + 1}/{epochs} | Value Loss: {epoch_v_loss / max(1, steps):.4f} | Policy Loss: {epoch_p_loss / max(1, steps):.4f}")

    checkpoint_path = "checkpoints/glaubermon_max_latest.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nTraining Complete! Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    train_glaubermon(num_games=10, epochs=3)
