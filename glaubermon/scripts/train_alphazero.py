"""AlphaZero / ReBeL Continuous Self-Play Reinforcement Learning Engine for Glaubermon Max.

Features:
- Pure self-play without relying on static datasets.
- Checkpoints every 10 games with automatic resume upon restart.
- Graceful Ctrl+C handling (saves instantly before exiting).
- Diverse Gen 9 OU competitive team pool with full ShowdownDex mechanics.
- Periodic evaluation against baseline to track win rate progression.
- Optimized for NVIDIA GeForce RTX 4090 with PyTorch AMP FP16.
"""

import os
import sys
import json
import time
import signal
import random
import argparse
from collections import deque
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, Hazard
from glaubermon.models.embeddings import encode_battle_state
from glaubermon.models.set_transformer import GlaubermonMaxNet
from glaubermon.search.subgame_resolver import SubgameResolver, simulate_turn_transition
from glaubermon.search.evaluators import NeuralEvaluator, HeuristicEvaluator, HybridEvaluator
from glaubermon.core.actions import action_to_logit_index
from glaubermon.data.showdown_dex import ShowdownDex


# --- Competitive Gen 9 OU Meta Pool ---
COMPETITIVE_POOL = [
    ("Great Tusk", "Ground", "Fighting", ["Close Combat", "Headlong Rush", "Ice Spinner", "Rapid Spin"]),
    ("Dragapult", "Dragon", "Ghost", ["Shadow Ball", "Draco Meteor", "Flamethrower", "U-turn"]),
    ("Kingambit", "Dark", "Steel", ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Swords Dance"]),
    ("Gholdengo", "Steel", "Ghost", ["Make It Rain", "Shadow Ball", "Focus Blast", "Nasty Plot"]),
    ("Dondozo", "Water", None, ["Liquidation", "Body Press", "Rest", "Sleep Talk"]),
    ("Iron Valiant", "Fairy", "Fighting", ["Moonblast", "Close Combat", "Knock Off", "Thunderbolt"]),
    ("Ting-Lu", "Dark", "Ground", ["Earthquake", "Ruination", "Stealth Rock", "Whirlwind"]),
    ("Ogerpon-Wellspring", "Water", "Grass", ["Ivy Cudgel", "Horn Leech", "Play Rough", "Spiky Shield"]),
    ("Gliscor", "Ground", "Flying", ["Earthquake", "Toxic", "Protect", "Spikes"]),
    ("Rillaboom", "Grass", None, ["Grassy Glide", "Wood Hammer", "Knock Off", "U-turn"]),
    ("Corviknight", "Flying", "Steel", ["Brave Bird", "Body Press", "Roost", "Defog"]),
    ("Heatran", "Fire", "Steel", ["Magma Storm", "Earth Power", "Flash Cannon", "Stealth Rock"]),
    ("Zamazenta", "Fighting", None, ["Close Combat", "Body Press", "Iron Defense", "Crunch"]),
    ("Samurott-Hisui", "Water", "Dark", ["Ceaseless Edge", "Razor Shell", "Knock Off", "Sucker Punch"]),
    ("Roaring Moon", "Dragon", "Dark", ["Knock Off", "Dragon Dance", "Earthquake", "Acrobatics"]),
]


def generate_competitive_battle() -> BattleState:
    """Generate a high-level competitive 6v6 match from meta pool."""
    from glaubermon.data.meta_teams import get_meta_team_balance, get_meta_team_hyper_offense, get_meta_team_stall, build_meta_pokemon
    dex = ShowdownDex.get_instance()

    # 50% chance of official tournament team archetype, 50% chance of random 6-mon meta draft
    if random.random() < 0.5:
        teams = [get_meta_team_balance, get_meta_team_hyper_offense, get_meta_team_stall]
        t1_gen = random.choice(teams)
        t2_gen = random.choice(teams)
        return BattleState(
            p1=BattleSide(pokemon=t1_gen(), hazards={Hazard.STEALTH_ROCK: 1} if random.random() < 0.5 else {}),
            p2=BattleSide(pokemon=t2_gen(), hazards={Hazard.STEALTH_ROCK: 1} if random.random() < 0.5 else {})
        )

    p1_picks = random.sample(COMPETITIVE_POOL, 6)
    p2_picks = random.sample(COMPETITIVE_POOL, 6)

    def build_team(picks):
        return [build_meta_pokemon(name, move_names) for name, t1_str, t2_str, move_names in picks]

    return BattleState(
        p1=BattleSide(pokemon=build_team(p1_picks), hazards={Hazard.STEALTH_ROCK: 1}),
        p2=BattleSide(pokemon=build_team(p2_picks), hazards={Hazard.STEALTH_ROCK: 1})
    )


class AlphaZeroReplayBuffer:
    """Sliding-window experience replay buffer for self-play transitions."""

    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_tensors, target_policy, outcome):
        self.buffer.append((state_tensors, target_policy, outcome))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class AlphaZeroTrainer:
    """Orchestrates Self-Play, MCTS Subgame Search, and Continuous GPU Optimization."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        d_model: int = 256,
        nhead: int = 8,
        lr: float = 3e-4,
        device: Optional[torch.device] = None,
        reset_from_scratch: bool = False
    ):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.latest_ckpt = os.path.join(self.checkpoint_dir, "glaubermon_alphazero_latest.pt")
        self.meta_path = os.path.join(self.checkpoint_dir, "alphazero_meta.json")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AlphaZero / ReBeL Engine Initialized on: {self.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

        # 1. Initialize Model
        self.model = GlaubermonMaxNet(d_model=d_model, nhead=nhead, num_actions=14).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.CrossEntropyLoss()

        self.replay_buffer = AlphaZeroReplayBuffer(capacity=25000)
        self.neural_eval = NeuralEvaluator(self.model, self.device)
        self.evaluator = HybridEvaluator(self.neural_eval, HeuristicEvaluator(), weight_neural=0.60)
        self.resolver = SubgameResolver(evaluator=self.evaluator)

        # Meta tracking
        self.total_games = 0
        self.total_turns = 0
        self.start_time = time.time()
        self.elapsed_offset = 0.0

        # Load existing state if available and not resetting from scratch
        if not reset_from_scratch:
            self._load_checkpoint_if_exists()
        else:
            print("Reset flag active: starting training from scratch with clean weights.")

        # Graceful shutdown handler
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _load_checkpoint_if_exists(self):
        """Resume automatically from latest checkpoint if available."""
        if os.path.exists(self.latest_ckpt):
            print(f"Loading checkpoint from {self.latest_ckpt}...")
            try:
                state_dict = torch.load(self.latest_ckpt, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("Model weights successfully loaded.")
            except Exception as e:
                print(f"Could not load weights from {self.latest_ckpt}: {e}")
        elif os.path.exists("checkpoints/glaubermon_max_elite.pt"):
            print("Warm-starting from grandmaster pretrained weights (checkpoints/glaubermon_max_elite.pt)...")
            try:
                state_dict = torch.load("checkpoints/glaubermon_max_elite.pt", map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("Warm-start successful.")
            except Exception as e:
                print(f"Could not load grandmaster weights: {e}")
        elif os.path.exists("checkpoints/glaubermon_max_latest.pt"):
            print("Warm-starting from grandmaster pretrained weights (checkpoints/glaubermon_max_latest.pt)...")
            try:
                state_dict = torch.load("checkpoints/glaubermon_max_latest.pt", map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("Warm-start successful.")
            except Exception as e:
                print(f"Could not load grandmaster weights: {e}")

        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                    self.total_games = meta.get("total_games", 0)
                    self.total_turns = meta.get("total_turns", 0)
                    self.elapsed_offset = meta.get("elapsed_time_seconds", 0.0)
                    print(f"Resuming training: {self.total_games} games previously played.")
            except Exception:
                pass

    def save_checkpoint(self):
        """Save network weights and metadata tracker."""
        torch.save(self.model.state_dict(), self.latest_ckpt)
        meta = {
            "total_games": self.total_games,
            "total_turns": self.total_turns,
            "elapsed_time_seconds": self.elapsed_offset + (time.time() - self.start_time),
            "last_saved": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] Auto-saved checkpoint ({self.total_games} total games).")

    def _handle_interrupt(self, sig, frame):
        print("\nInterrupt signal received! Saving current model before exiting...")
        self.save_checkpoint()
        print("Model saved safely. Exiting cleanly.")
        sys.exit(0)

    def play_self_play_game(self) -> List[Tuple]:
        """Play one full self-play match using SubgameResolver search."""
        state = generate_competitive_battle()
        trajectory = []

        for turn in range(35):
            if state.is_game_over:
                break

            state_tensors = encode_battle_state(state)

            # Solve for P1
            act1, p1_strat, actions1, _ = self.resolver.resolve_turn(state, depth=1, sample=True)

            # Solve for P2 (perspective inverted)
            inv_state = BattleState(p1=state.p2, p2=state.p1, weather=state.weather, terrain=state.terrain, turn=state.turn)
            act2, _, _, _ = self.resolver.resolve_turn(inv_state, depth=1, sample=True)

            # Encode search policy distribution into exact 14-dim action logit targets
            target_policy = np.zeros(14, dtype=np.float32)
            for a, p in zip(actions1, p1_strat):
                idx = action_to_logit_index(a)
                target_policy[idx] += p

            trajectory.append((state_tensors, target_policy))

            # Execute turn
            state = simulate_turn_transition(state, act1, act2)

            # Force replacement if fainted
            if state.p1.active_pokemon.is_fainted and not state.p1.is_all_fainted:
                sw = state.p1.available_switches()
                if sw:
                    state.p1.active_index = sw[0]
            if state.p2.active_pokemon.is_fainted and not state.p2.is_all_fainted:
                sw = state.p2.available_switches()
                if sw:
                    state.p2.active_index = sw[0]

        if state.winner == 1:
            outcome = 1.0
        elif state.winner == 2:
            outcome = -1.0
        else:
            # Domain-grounded terminal score on turn cap (material, HP, hazards)
            outcome = float(HeuristicEvaluator().evaluate(state))

        # Pair trajectory states with outcome
        game_data = []
        for state_t, target_policy in trajectory:
            game_data.append((state_t, target_policy, outcome))
        return game_data

    def train_step(self, batch_size: int = 256) -> Tuple[float, float]:
        """Sample from replay buffer and run optimization step."""
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0

        batch = self.replay_buffer.sample(batch_size)
        self.model.train()

        p1_m_list, p1_s_list, p2_m_list, p2_s_list, f_list = [], [], [], [], []
        val_targets, policy_targets = [], []

        for state_tensors, target_policy, outcome in batch:
            (p1_m, p1_s), (p2_m, p2_s), field = state_tensors
            p1_m_list.append(p1_m)
            p1_s_list.append(p1_s)
            p2_m_list.append(p2_m)
            p2_s_list.append(p2_s)
            f_list.append(field)
            val_targets.append(outcome)
            policy_targets.append(torch.tensor(target_policy, dtype=torch.float32))

        p1_m_b = torch.stack(p1_m_list).to(self.device, non_blocking=True)
        p1_s_b = torch.stack(p1_s_list).to(self.device, non_blocking=True)
        p2_m_b = torch.stack(p2_m_list).to(self.device, non_blocking=True)
        p2_s_b = torch.stack(p2_s_list).to(self.device, non_blocking=True)
        f_b = torch.stack(f_list).to(self.device, non_blocking=True)
        v_targets_b = torch.tensor(val_targets, dtype=torch.float32, device=self.device).unsqueeze(-1)
        p_targets_b = torch.stack(policy_targets).to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            pred_v, pred_p = self.model(p1_m_b, p1_s_b, p2_m_b, p2_s_b, f_b)
            loss_v = self.value_loss_fn(pred_v, v_targets_b)
            loss_p = self.policy_loss_fn(pred_p, p_targets_b)
            loss = loss_v + loss_p

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return float(loss_v.item()), float(loss_p.item())

    def evaluate_vs_baseline(self, num_games: int = 5) -> float:
        """Benchmark current neural model against heuristic baseline player."""
        self.model.eval()
        heuristic = HeuristicEvaluator()
        wins = 0

        for _ in range(num_games):
            state = generate_competitive_battle()
            for turn in range(30):
                if state.is_game_over:
                    break
                act1, _, _, _ = self.resolver.resolve_turn(state, depth=1, sample=False)

                # Baseline heuristic action
                inv_state = BattleState(p1=state.p2, p2=state.p1, weather=state.weather, terrain=state.terrain, turn=state.turn)
                p2_acts = inv_state.get_valid_actions(player=1)
                best_act = p2_acts[0]
                best_val = -999.0
                for a in p2_acts:
                    sim = simulate_turn_transition(inv_state, a, None)
                    v = heuristic.evaluate(sim)
                    if v > best_val:
                        best_val = v
                        best_act = a

                state = simulate_turn_transition(state, act1, best_act)
                if state.p1.active_pokemon.is_fainted and not state.p1.is_all_fainted:
                    sw = state.p1.available_switches()
                    if sw:
                        state.p1.active_index = sw[0]
                if state.p2.active_pokemon.is_fainted and not state.p2.is_all_fainted:
                    sw = state.p2.available_switches()
                    if sw:
                        state.p2.active_index = sw[0]

            if state.winner == 1:
                wins += 1

        return wins / max(1, num_games)

    def train(self, games_to_play: int = 1000, save_every: int = 10, eval_every: int = 25):
        target_games = self.total_games + games_to_play
        print("=" * 75)
        print("  GLAUBERMON MAX: ALPHAZERO SELF-PLAY REINFORCEMENT LEARNING")
        print(f"  Playing {games_to_play} new self-play matches ({self.total_games} -> {target_games} total)")
        print(f"  Auto-saving every {save_every} games | Benchmarking every {eval_every} games")
        print("  Press Ctrl+C at any time to pause and save safely.")
        print("=" * 75)

        pbar = tqdm(total=target_games, initial=self.total_games, desc="AlphaZero Self-Play")

        while self.total_games < target_games:
            # 1. Play Self-Play Match
            samples = self.play_self_play_game()
            for s in samples:
                self.replay_buffer.push(*s)
            self.total_games += 1
            self.total_turns += len(samples)
            pbar.update(1)

            # 2. Train network on experience replay buffer
            if len(self.replay_buffer) >= 64:
                v_loss, p_loss = self.train_step(batch_size=min(256, len(self.replay_buffer)))
                pbar.set_postfix({"V-Loss": f"{v_loss:.3f}", "P-Loss": f"{p_loss:.3f}", "Buffer": len(self.replay_buffer)})

            # 3. Periodic Checkpointing
            if self.total_games % save_every == 0:
                self.save_checkpoint()

            # 4. Periodic Arena Benchmark vs Baseline
            if self.total_games % eval_every == 0:
                win_rate = self.evaluate_vs_baseline(num_games=4)
                print(f"\n[Arena Benchmark @ Game {self.total_games}] Winrate vs Heuristic Baseline: {win_rate * 100:.1f}%\n")

        pbar.close()
        self.save_checkpoint()
        print(f"\nTraining session complete! {self.total_games} total games recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glaubermon Max AlphaZero / ReBeL Self-Play Trainer")
    parser.add_argument("--games", type=int, default=500, help="Target total games to reach")
    parser.add_argument("--save-every", type=int, default=10, help="Save model every N games")
    parser.add_argument("--eval-every", type=int, default=25, help="Run arena evaluation every N games")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch without loading prior weights")
    args = parser.parse_args()

    trainer = AlphaZeroTrainer(reset_from_scratch=args.from_scratch)
    trainer.train(games_to_play=args.games, save_every=args.save_every, eval_every=args.eval_every)
