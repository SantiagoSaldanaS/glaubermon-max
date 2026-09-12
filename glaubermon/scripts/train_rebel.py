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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import time
import signal
import random
import argparse
import asyncio
import hashlib
from pathlib import Path
from collections import deque
from dataclasses import replace
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
from glaubermon.search.subgame_resolver import SubgameResolver, simulate_turn_transition, apply_entry_hazards
from glaubermon.search.evaluators import NeuralEvaluator, HeuristicEvaluator, HybridEvaluator
from glaubermon.scripts.mass_battle_auditor import execute_showdown_accurate_force_switch
from glaubermon.core.actions import action_to_logit_index, SwitchAction
from glaubermon.data.showdown_dex import ShowdownDex
from glaubermon.data.meta_teams import (
    get_meta_team_balance,
    get_meta_team_hyper_offense,
    get_meta_team_stall,
    get_meta_team_pelol94,
    build_meta_pokemon
)


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
    dex = ShowdownDex.get_instance()

    # 50% chance of official tournament team archetype, 50% chance of random 6-mon meta draft
    if random.random() < 0.5:
        teams = [get_meta_team_balance, get_meta_team_hyper_offense, get_meta_team_stall, get_meta_team_pelol94]
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
        checkpoint_dir: str = "runs/rebel-official-v2",
        d_model: int = 256,
        nhead: int = 8,
        lr: float = 3e-4,
        device: Optional[torch.device] = None,
        reset_from_scratch: bool = False,
        mechanics_seed: int = 0,
        rollout_backend: str = "showdown",
        showdown_path: Optional[str] = None,
        max_turns: int = 300,
        depth: int = 1,
    ):
        if rollout_backend not in ("showdown", "internal-privileged"):
            raise ValueError("Unknown rollout backend")
        if max_turns < 1 or depth < 1:
            raise ValueError("max_turns and depth must be positive")
        self.rollout_backend = rollout_backend
        self.rollout_mode = "official_public_v2" if rollout_backend == "showdown" else "sampled_internal_privileged_v2"
        project_root = Path(__file__).resolve().parents[2]
        local_showdown = project_root / 'tools/showdown/node_modules/pokemon-showdown'
        legacy_showdown = project_root.parent / 'showdown-parity/node_modules/pokemon-showdown'
        self.showdown_path = str(Path(showdown_path).resolve()) if showdown_path else str(local_showdown if local_showdown.exists() else legacy_showdown)
        if rollout_backend == 'showdown' and not Path(self.showdown_path, 'package.json').exists():
            raise FileNotFoundError("Install the pinned official simulator and set --showdown-path before training")
        self.max_turns = max_turns
        self.depth = depth
        self.last_game_info = {}
        random.seed(mechanics_seed)
        np.random.seed(mechanics_seed)
        torch.manual_seed(mechanics_seed)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.latest_ckpt = os.path.join(self.checkpoint_dir, "glaubermon_rebel_latest.pt")
        self.meta_path = os.path.join(self.checkpoint_dir, "rebel_meta.json")
        self.mechanics_seed = mechanics_seed

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AlphaZero / ReBeL Engine Initialized on: {self.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

        # 1. Initialize Model
        self.model = GlaubermonMaxNet(d_model=d_model, nhead=nhead, num_actions=14).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.CrossEntropyLoss()

        self.replay_buffer = AlphaZeroReplayBuffer(capacity=25000)
        self.evaluator = HybridEvaluator(NeuralEvaluator(self.model, self.device), HeuristicEvaluator(), weight_neural=0.60)
        self.resolver = SubgameResolver(evaluator=self.evaluator)

        # Meta tracking
        self.total_games = 0
        self.total_turns = 0
        self.total_samples = 0
        self.start_time = time.time()
        self.elapsed_offset = 0.0

        # Load existing state if available and not resetting from scratch
        if not reset_from_scratch:
            self._load_checkpoint_if_exists()
        else:
            print("Reset flag active: starting training from scratch with clean weights.")

        # Graceful shutdown handler
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _data_contract(self):
        contract = dict(rollout_mode=self.rollout_mode, depth=self.depth, max_turns=self.max_turns,
                        value_target="terminal_only", observation_version="public_restrictions_v5")
        root = Path(__file__).resolve().parents[1]
        contract["encoder_sha256"] = hashlib.sha256((root/'models/embeddings.py').read_bytes()).hexdigest()
        if self.rollout_backend == "showdown":
            contract["observer_sha256"] = hashlib.sha256((root/'client/showdown_bot.py').read_bytes()).hexdigest()
            contract["bridge_sha256"] = hashlib.sha256((root/'evaluation/showdown_bridge.cjs').read_bytes()).hexdigest()
        if self.rollout_backend == "showdown":
            from glaubermon.evaluation.training_teams import TRAINING_TEAMS
            contract["showdown_version"] = json.loads(Path(self.showdown_path, 'package.json').read_text())['version']
            contract["training_pool_sha256"] = hashlib.sha256(json.dumps(TRAINING_TEAMS,sort_keys=True).encode()).hexdigest()
        return contract

    def _load_checkpoint_if_exists(self):
        """Load weights explicitly; incompatible files must not silently train a random net."""
        candidates = [self.latest_ckpt, "checkpoints/glaubermon_rebel_latest.pt",
                      "checkpoints/glaubermon_max_elite.pt", "checkpoints/glaubermon_max_latest.pt"]
        source = next((path for path in candidates if os.path.exists(path)), None)
        if os.path.exists(self.meta_path) and not os.path.exists(self.latest_ckpt):
            raise FileNotFoundError(f"Metadata exists without its checkpoint: {self.latest_ckpt}")
        if source is None:
            print("No checkpoint found: using the newly initialized model.")
            return
        self.model.load_compatible_state_dict(torch.load(source, map_location=self.device, weights_only=True))
        print(f"Loaded model weights from {source}")
        if source != self.latest_ckpt:
            print("New experiment: original training counters are not copied.")
            return
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError("Output checkpoint has no provenance metadata; choose a new output directory")
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("rollout_mode") != self.rollout_mode:
                raise ValueError("Checkpoint counters use another rollout mode; choose a new --checkpoint-dir.")
            if meta.get("data_contract") != self._data_contract():
                raise ValueError("Training data contract changed; choose a new --checkpoint-dir")
            self.total_games = meta.get("total_games", 0)
            self.total_samples = meta.get("total_samples", 0)
            self.total_turns = meta.get("total_turns", 0)
            self.elapsed_offset = meta.get("elapsed_time_seconds", 0.0)
            self.mechanics_seed = meta.get("mechanics_seed", self.mechanics_seed)
            print(f"Restored weights/counters ({self.total_games} games); optimizer and replay buffer restart.")

    def save_checkpoint(self):
        """Save network weights and metadata tracker."""
        torch.save(self.model.state_dict(), self.latest_ckpt)
        if self.total_games > 0 and self.total_games % 1000 == 0:
            milestone_path = os.path.join(self.checkpoint_dir, f"glaubermon_rebel_{self.total_games}g.pt")
            torch.save(self.model.state_dict(), milestone_path)
        meta = {
            "rollout_mode": self.rollout_mode,
            "data_contract": self._data_contract(),
            "total_samples": self.total_samples,
            "rollout_backend": self.rollout_backend,
            "observation_source": "live_ShowdownBot_player_channel" if self.rollout_backend == "showdown" else "privileged_internal_truth_experimental",
            "truncation_value_target": "missing_masked",
            "max_turns": self.max_turns,
            "depth": self.depth,
            "showdown_version": json.loads(Path(self.showdown_path, 'package.json').read_text())['version'] if self.rollout_backend == 'showdown' else None,
            "last_game": self.last_game_info,
            "torch_version": torch.__version__,
            "device": str(self.device),
            "torch_threads": torch.get_num_threads(),
            "evaluator": "hybrid_0.60",
            "mechanics_seed": self.mechanics_seed,
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

    def _replace_fainted(self, state: BattleState, resolver: SubgameResolver, opponent_resolver=None, rng=None):
        """Resolve every requested replacement and resume any queued turn actions."""
        if not state.pending_switches:
            state.pending_switches = tuple(i for i,side in ((1,state.p1),(2,state.p2))
                if side.active_pokemon.is_fainted and not side.is_all_fainted)
        while state.pending_switches and not state.is_game_over:
            actions = [None,None]
            for side_idx in state.pending_switches:
                side = state.p1 if side_idx == 1 else state.p2
                active_resolver = opponent_resolver if side_idx == 2 and opponent_resolver is not None else resolver
                if resolver is None:  # Explicit replacement policy supplied by a test/caller.
                    view = state if side_idx == 1 else state.flipped()
                    slot = execute_showdown_accurate_force_switch(active_resolver,side,view.p2.active_pokemon,view)
                    actions[side_idx-1] = SwitchAction(slot+1,side.pokemon[slot].species)
                else:
                    result = active_resolver.resolve_turn(state,depth=1,sample=True,return_both_players=True)
                    actions[side_idx-1] = result[0 if side_idx == 1 else 4]
            after = simulate_turn_transition(state,*actions,sample_outcomes=True,rng=rng)
            state.__dict__.update(after.__dict__)

    def play_self_play_game(self) -> List[Tuple]:
        """Play one full self-play match using SubgameResolver search."""
        if self.rollout_backend == "showdown":
            from glaubermon.evaluation.official_self_play import collect_game
            seed = self.mechanics_seed + self.total_games
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.model.eval()
            # Separate training pool. Frozen benchmark teams must not be used here.
            from glaubermon.evaluation.training_teams import TRAINING_TEAMS
            keys = list(TRAINING_TEAMS)
            team_rng = random.Random(seed)
            teams = [team_rng.choice(keys), team_rng.choice(keys)]
            trace_dir = Path(self.checkpoint_dir) / 'rollouts'
            trace_dir.mkdir(exist_ok=True)
            trace = trace_dir / f"game-{self.total_games:06d}.jsonl"
            samples, self.last_game_info = asyncio.run(collect_game(
                self.model, self.showdown_path, teams, [seed % 65536, 19283, 38475, 29384],
                depth=self.depth, max_turns=self.max_turns, trace_path=trace,
                team_pool=TRAINING_TEAMS))
            return samples
        state = generate_competitive_battle()
        trajectory = []
        self.model.eval()  # train_step enables dropout; search must disable it again.
        rng = random.Random(self.mechanics_seed + self.total_games)

        for turn in range(self.max_turns):
            if state.is_game_over:
                break

            state_tensors = encode_battle_state(state)

            # Solve for P1
            act1, p1_strat, actions1, _ = self.resolver.resolve_turn(state, depth=self.depth, sample=True)

            # Solve for P2 (perspective inverted)
            inv_state = state.flipped()
            act2, _, _, _ = self.resolver.resolve_turn(inv_state, depth=self.depth, sample=True)

            # Encode search policy distribution into exact 14-dim action logit targets
            target_policy = np.zeros(14, dtype=np.float32)
            for a, p in zip(actions1, p1_strat):
                idx = action_to_logit_index(a)
                target_policy[idx] += p

            trajectory.append((state_tensors, target_policy))

            # Execute turn
            state = simulate_turn_transition(state, act1, act2, sample_outcomes=True, rng=rng)

            # Force replacement if fainted
            self._replace_fainted(state, self.resolver, rng=rng)

        if state.winner == 1:
            outcome = 1.0
        elif state.winner == 2:
            outcome = -1.0
        else:
            outcome = 0.0 if state.is_game_over else None
        self.last_game_info = dict(terminated=state.is_game_over, winner=state.winner,
                                   turns=len(trajectory), samples=len(trajectory))

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
        val_targets, policy_targets, value_mask = [], [], []

        for state_tensors, target_policy, outcome in batch:
            (p1_m, p1_s), (p2_m, p2_s), field = state_tensors
            p1_m_list.append(p1_m)
            p1_s_list.append(p1_s)
            p2_m_list.append(p2_m)
            p2_s_list.append(p2_s)
            f_list.append(field)
            value_mask.append(outcome is not None)
            val_targets.append(0.0 if outcome is None else outcome)
            policy_targets.append(torch.tensor(target_policy, dtype=torch.float32))

        p1_m_b = torch.stack(p1_m_list).to(self.device, non_blocking=True)
        p1_s_b = torch.stack(p1_s_list).to(self.device, non_blocking=True)
        p2_m_b = torch.stack(p2_m_list).to(self.device, non_blocking=True)
        p2_s_b = torch.stack(p2_s_list).to(self.device, non_blocking=True)
        f_b = torch.stack(f_list).to(self.device, non_blocking=True)
        v_targets_b = torch.tensor(val_targets, dtype=torch.float32, device=self.device).unsqueeze(-1)
        p_targets_b = torch.stack(policy_targets).to(self.device, non_blocking=True)

        v_mask = torch.tensor(value_mask, dtype=torch.bool, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            pred_v, pred_p = self.model(p1_m_b, p1_s_b, p2_m_b, p2_s_b, f_b)
            # No value-head update (including weight decay) on wholly truncated batches.
            loss_v = self.value_loss_fn(pred_v[v_mask], v_targets_b[v_mask]) if v_mask.any() else pred_p.new_zeros(())
            loss_p = self.policy_loss_fn(pred_p, p_targets_b)
            loss = loss_v + loss_p

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return float(loss_v.item()), float(loss_p.item())

    def evaluate_vs_baseline(self, num_games: int = 5) -> float:
        """Benchmark current neural model against heuristic baseline player."""
        self.model.eval()
        neural_resolver = SubgameResolver(evaluator=self.evaluator)
        heuristic = HeuristicEvaluator()
        heuristic_resolver = SubgameResolver(evaluator=heuristic)
        wins = 0

        for game_idx in range(num_games):
            state = generate_competitive_battle()
            rng = random.Random(self.mechanics_seed + 1_000_000 + game_idx)
            for turn in range(30):
                if state.is_game_over:
                    break
                act1, _, _, _ = neural_resolver.resolve_turn(state, depth=1, sample=False)

                # Baseline heuristic action
                inv_state = state.flipped()
                p2_acts = inv_state.get_valid_actions(player=1)
                best_act = p2_acts[0]
                best_val = -999.0
                for a in p2_acts:
                    sim = simulate_turn_transition(inv_state, a, None)
                    v = heuristic.evaluate(sim)
                    if v > best_val:
                        best_val = v
                        best_act = a

                state = simulate_turn_transition(state, act1, best_act, sample_outcomes=True, rng=rng)
                self._replace_fainted(state, neural_resolver, heuristic_resolver)

            if state.winner == 1:
                wins += 1

        return wins / max(1, num_games)

    def train(self, games_to_play: int = 1000, save_every: int = 10, eval_every: int = 0):
        alignment = Path(__file__).resolve().parents[2] / "docs/ALIGNMENT_STATUS.json"
        if not alignment.exists() or not json.loads(alignment.read_text()).get("training_allowed",False):
            raise RuntimeError("Training paused pending environment alignment; see docs/ALIGNMENT_STATUS.json")
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
            self.total_samples += len(samples)
            self.total_turns += self.last_game_info.get("turns", len(samples))
            pbar.update(1)

            # 2. Train network on experience replay buffer
            if len(self.replay_buffer) >= 64:
                v_loss, p_loss = self.train_step(batch_size=min(256, len(self.replay_buffer)))
                pbar.set_postfix({"V-Loss": f"{v_loss:.3f}", "P-Loss": f"{p_loss:.3f}", "Buffer": len(self.replay_buffer)})

            # 3. Periodic Checkpointing
            if self.total_games % save_every == 0:
                self.save_checkpoint()

            # 4. Periodic Arena Benchmark vs Baseline
            if eval_every and self.rollout_backend == "internal-privileged" and self.total_games % eval_every == 0:
                win_rate = self.evaluate_vs_baseline(num_games=4)
                print(f"\n[Arena Benchmark @ Game {self.total_games}] Winrate vs Heuristic Baseline: {win_rate * 100:.1f}%\n")

        pbar.close()
        self.save_checkpoint()
        print(f"\nTraining session complete! {self.total_games} total games recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glaubermon Max AlphaZero / ReBeL Self-Play Trainer")
    parser.add_argument("--games", type=int, default=500, help="Target total games to reach")
    parser.add_argument("--save-every", type=int, default=10, help="Save model every N games")
    parser.add_argument("--eval-every", type=int, default=0, help="Experimental internal diagnostic only; official evaluation is a separate frozen run")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch without loading prior weights")
    parser.add_argument("--checkpoint-dir", default="runs/rebel-official-v2", help="Separate output directory; original checkpoints stay intact")
    parser.add_argument("--mechanics-seed", type=int, default=0, help="Seed for official mechanics, teams, policy sampling and model initialization")
    parser.add_argument("--rollout-backend", choices=["showdown", "internal-privileged"], default="showdown")
    parser.add_argument("--showdown-path", help="Path to pinned pokemon-showdown npm package")
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU threads for small inference batches; recorded in checkpoint metadata")
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)

    trainer = AlphaZeroTrainer(checkpoint_dir=args.checkpoint_dir, reset_from_scratch=args.from_scratch,
                              mechanics_seed=args.mechanics_seed, rollout_backend=args.rollout_backend,
                              showdown_path=args.showdown_path, max_turns=args.max_turns, depth=args.depth)
    trainer.train(games_to_play=args.games, save_every=args.save_every, eval_every=args.eval_every)
