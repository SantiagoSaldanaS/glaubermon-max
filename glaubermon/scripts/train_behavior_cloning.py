"""Behavior Cloning (Supervised Pretraining) on High-Elo Showdown Replays."""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from glaubermon.data.replay_parser import ReplayParser
from glaubermon.models.set_transformer import GlaubermonMaxNet


class ShowdownExpertDataset(Dataset):
    """PyTorch Dataset wrapping parsed human battle states and actions."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # sample: (state_tensors, action_idx, outcome)
        # state_tensors: ((p1_m, p1_s), (p2_m, p2_s), field)
        state_tensors, action_idx, outcome = self.samples[idx]
        (p1_m, p1_s), (p2_m, p2_s), field = state_tensors
        return (
            p1_m,
            p1_s,
            p2_m,
            p2_s,
            field,
            torch.tensor(action_idx, dtype=torch.long),
            torch.tensor(outcome, dtype=torch.float32)
        )


def train_behavior_cloning(
    replays_dir: str = "data/replays",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    checkpoint_out: str = "checkpoints/glaubermon_max_latest.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  GLAUBERMON MAX: SUPERVISED BEHAVIOR CLONING ON HIGH-ELO REPLAYS")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    # 1. Parse all downloaded replays
    parser = ReplayParser(replays_dir=replays_dir)
    raw_samples = parser.parse_all()

    if not raw_samples:
        print("No samples found! Please run the harvester first.")
        return

    # Shuffle and split train / val (90% train, 10% val)
    random.seed(42)
    random.shuffle(raw_samples)
    split_idx = int(len(raw_samples) * 0.9)
    train_samples = raw_samples[:split_idx]
    val_samples = raw_samples[split_idx:]

    print(f"\nDataset size: {len(raw_samples)} total states ({len(train_samples)} train, {len(val_samples)} validation)")

    train_loader = DataLoader(ShowdownExpertDataset(train_samples), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ShowdownExpertDataset(val_samples), batch_size=batch_size, shuffle=False)

    # 2. Initialize Model
    model = GlaubermonMaxNet(d_model=128, nhead=4, num_actions=14).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    best_val_acc = 0.0

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_p_loss = 0.0
        train_v_loss = 0.0
        correct_top1 = 0
        correct_top3 = 0
        total = 0

        for p1_m, p1_s, p2_m, p2_s, field, target_act, target_val in train_loader:
            p1_m = p1_m.to(device)
            p1_s = p1_s.to(device)
            p2_m = p2_m.to(device)
            p2_s = p2_s.to(device)
            field = field.to(device)
            target_act = target_act.to(device)
            target_val = target_val.to(device).unsqueeze(-1)

            optimizer.zero_grad()
            pred_v, pred_p = model(p1_m, p1_s, p2_m, p2_s, field)

            loss_p = policy_criterion(pred_p, target_act)
            loss_v = value_criterion(pred_v, target_val)
            loss = loss_p + 0.5 * loss_v

            loss.backward()
            optimizer.step()

            train_p_loss += loss_p.item() * len(target_act)
            train_v_loss += loss_v.item() * len(target_act)

            # Accuracy
            _, top1_preds = torch.max(pred_p, dim=-1)
            correct_top1 += (top1_preds == target_act).sum().item()

            top3_preds = torch.topk(pred_p, k=min(3, pred_p.shape[-1]), dim=-1).indices
            correct_top3 += (top3_preds == target_act.unsqueeze(-1)).any(dim=-1).sum().item()

            total += len(target_act)

        train_p_loss /= max(1, total)
        train_v_loss /= max(1, total)
        train_top1 = correct_top1 / max(1, total)
        train_top3 = correct_top3 / max(1, total)

        # Validation phase
        model.eval()
        val_correct_top1 = 0
        val_correct_top3 = 0
        val_total = 0

        with torch.no_grad():
            for p1_m, p1_s, p2_m, p2_s, field, target_act, target_val in val_loader:
                p1_m = p1_m.to(device)
                p1_s = p1_s.to(device)
                p2_m = p2_m.to(device)
                p2_s = p2_s.to(device)
                field = field.to(device)
                target_act = target_act.to(device)

                _, pred_p = model(p1_m, p1_s, p2_m, p2_s, field)

                _, top1_preds = torch.max(pred_p, dim=-1)
                val_correct_top1 += (top1_preds == target_act).sum().item()

                top3_preds = torch.topk(pred_p, k=min(3, pred_p.shape[-1]), dim=-1).indices
                val_correct_top3 += (top3_preds == target_act.unsqueeze(-1)).any(dim=-1).sum().item()

                val_total += len(target_act)

        val_top1 = val_correct_top1 / max(1, val_total)
        val_top3 = val_correct_top3 / max(1, val_total)

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Policy Loss: {train_p_loss:.4f} | Value Loss: {train_v_loss:.4f} | "
            f"Train Top-1: {train_top1 * 100:.1f}% | Train Top-3: {train_top3 * 100:.1f}% | "
            f"Val Top-1: {val_top1 * 100:.1f}% | Val Top-3: {val_top3 * 100:.1f}%"
        )

        # Save best checkpoint
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            os.makedirs(os.path.dirname(checkpoint_out), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_out)
            torch.save(model.state_dict(), "checkpoints/glaubermon_max_bc.pt")

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Saved optimal pretrained weights to: {checkpoint_out}")


if __name__ == "__main__":
    train_behavior_cloning(epochs=8, batch_size=64)
