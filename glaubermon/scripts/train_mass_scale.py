"""Mass-Scale Grandmaster Pretraining on 100,000+ Elite Showdown Turns (RTX 4090 Optimized)."""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from glaubermon.data.mass_dataset import load_mass_dataset
from glaubermon.scripts.train_behavior_cloning import ShowdownExpertDataset
from glaubermon.models.set_transformer import GlaubermonMaxNet


def train_mass_scale(
    part_numbers=[50, 80],
    min_rating: int = 1500,
    max_games: int = 15000,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 5e-4,
    d_model: int = 256,
    nhead: int = 8,
    checkpoint_out: str = "checkpoints/glaubermon_max_latest.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("  GLAUBERMON MAX: MASS-SCALE GRANDMASTER TRAINING (RTX 4090)")
    print(f"  Target Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM Available:  {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if torch.cuda.is_available() else "")
    print("=" * 75)

    # 1. Load Grandmaster Dataset
    raw_samples = load_mass_dataset(
        part_numbers=part_numbers,
        min_rating=min_rating,
        max_games=max_games
    )

    if not raw_samples:
        print("No samples loaded!")
        return

    random.seed(42)
    random.shuffle(raw_samples)
    split_idx = int(len(raw_samples) * 0.92)
    train_samples = raw_samples[:split_idx]
    val_samples = raw_samples[split_idx:]

    print(f"\nExtracted {len(raw_samples)} Grandmaster Decision States:")
    print(f"  • Training States:   {len(train_samples):,}")
    print(f"  • Validation States: {len(val_samples):,}")

    train_loader = DataLoader(
        ShowdownExpertDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    val_loader = DataLoader(
        ShowdownExpertDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    # 2. Initialize Model with Scaled Capacity
    model = GlaubermonMaxNet(d_model=d_model, nhead=nhead, num_actions=14).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    best_val_top1 = 0.0

    print(f"\nStarting optimization across {epochs} epochs (Batch Size: {batch_size}, AMP FP16 enabled)...")

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_p_loss = 0.0
        train_v_loss = 0.0
        correct_top1 = 0
        correct_top3 = 0
        total = 0

        for p1_m, p1_s, p2_m, p2_s, field, target_act, target_val in train_loader:
            p1_m = p1_m.to(device, non_blocking=True)
            p1_s = p1_s.to(device, non_blocking=True)
            p2_m = p2_m.to(device, non_blocking=True)
            p2_s = p2_s.to(device, non_blocking=True)
            field = field.to(device, non_blocking=True)
            target_act = target_act.to(device, non_blocking=True)
            target_val = target_val.to(device, non_blocking=True).unsqueeze(-1)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                pred_v, pred_p = model(p1_m, p1_s, p2_m, p2_s, field)
                loss_p = policy_criterion(pred_p, target_act)
                loss_v = value_criterion(pred_v, target_val)
                loss = loss_p + 0.5 * loss_v

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = len(target_act)
            train_p_loss += loss_p.item() * bs
            train_v_loss += loss_v.item() * bs

            _, top1_preds = torch.max(pred_p, dim=-1)
            correct_top1 += (top1_preds == target_act).sum().item()

            top3_preds = torch.topk(pred_p, k=min(3, pred_p.shape[-1]), dim=-1).indices
            correct_top3 += (top3_preds == target_act.unsqueeze(-1)).any(dim=-1).sum().item()

            total += bs

        scheduler.step()

        train_p_loss /= max(1, total)
        train_v_loss /= max(1, total)
        train_top1 = correct_top1 / max(1, total)
        train_top3 = correct_top3 / max(1, total)

        # Validation Phase
        model.eval()
        val_correct_top1 = 0
        val_correct_top3 = 0
        val_total = 0

        with torch.no_grad():
            for p1_m, p1_s, p2_m, p2_s, field, target_act, target_val in val_loader:
                p1_m = p1_m.to(device, non_blocking=True)
                p1_s = p1_s.to(device, non_blocking=True)
                p2_m = p2_m.to(device, non_blocking=True)
                p2_s = p2_s.to(device, non_blocking=True)
                field = field.to(device, non_blocking=True)
                target_act = target_act.to(device, non_blocking=True)

                with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    _, pred_p = model(p1_m, p1_s, p2_m, p2_s, field)

                bs = len(target_act)
                _, top1_preds = torch.max(pred_p, dim=-1)
                val_correct_top1 += (top1_preds == target_act).sum().item()

                top3_preds = torch.topk(pred_p, k=min(3, pred_p.shape[-1]), dim=-1).indices
                val_correct_top3 += (top3_preds == target_act.unsqueeze(-1)).any(dim=-1).sum().item()

                val_total += bs

        val_top1 = val_correct_top1 / max(1, val_total)
        val_top3 = val_correct_top3 / max(1, val_total)

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"P-Loss: {train_p_loss:.4f} | V-Loss: {train_v_loss:.4f} | "
            f"Train Top-1: {train_top1 * 100:.1f}% | Train Top-3: {train_top3 * 100:.1f}% | "
            f"Val Top-1: {val_top1 * 100:.1f}% | Val Top-3: {val_top3 * 100:.1f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            os.makedirs(os.path.dirname(checkpoint_out), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_out)
            torch.save(model.state_dict(), "checkpoints/glaubermon_max_elite.pt")

    print(f"\nGrandmaster Training Complete! Peak Validation Top-1 Accuracy: {best_val_top1 * 100:.2f}%")
    print(f"Saved optimal weights to: {checkpoint_out}")


if __name__ == "__main__":
    train_mass_scale(
        part_numbers=[50, 80],
        min_rating=1500,
        max_games=15000,
        epochs=10,
        batch_size=512,
        d_model=256,
        nhead=8
    )
