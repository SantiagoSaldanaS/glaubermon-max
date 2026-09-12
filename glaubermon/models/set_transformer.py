"""Scaled Permutation-Invariant Set Transformer for Competitive Pokémon 6v6 Battles."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from glaubermon.models.embeddings import MOVE_DIM, STAT_DIM


class SetAttentionBlock(nn.Module):
    """Permutation-equivariant self-attention block for sets of tokens."""

    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, set_size, d_model)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + ff_out)
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block allowing Team 1 to attend to Team 2 threats."""

    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        x = self.norm1(q + attn_out)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm2(x + ff_out)


class GlaubermonMaxNet(nn.Module):
    """Deep Scaled Permutation-Invariant Dual-Head Actor-Critic Set Transformer."""

    def __init__(self, d_model: int = 256, nhead: int = 8, num_actions: int = 14):
        super().__init__()
        self.d_model = d_model
        move_emb_dim = 128

        # 1. Move Encoder: (batch, 6, 4, MOVE_DIM) -> (batch, 6, 128)
        self.move_fc = nn.Sequential(
            nn.Linear(MOVE_DIM, move_emb_dim),
            nn.GELU(),
            nn.Linear(move_emb_dim, move_emb_dim),
            nn.LayerNorm(move_emb_dim)
        )

        # 2. Pokémon Token Projector: (moves 128 + stats STAT_DIM) -> d_model (256)
        self.mon_projector = nn.Sequential(
            nn.Linear(move_emb_dim + STAT_DIM, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # 3. Team Set Attention (Intra-team synergies & team structures)
        self.team_sab1 = SetAttentionBlock(d_model, nhead=nhead, dim_feedforward=d_model * 2)
        self.team_sab2 = SetAttentionBlock(d_model, nhead=nhead, dim_feedforward=d_model * 2)
        self.team_sab3 = SetAttentionBlock(d_model, nhead=nhead, dim_feedforward=d_model * 2)

        # 4. Bidirectional Cross-Attention (P1 team vs P2 team matchups)
        self.cross_attn1 = CrossAttentionBlock(d_model, nhead=nhead, dim_feedforward=d_model * 2)
        self.cross_attn2 = CrossAttentionBlock(d_model, nhead=nhead, dim_feedforward=d_model * 2)

        # 5. Field Encoder: 40 public features -> d_model
        self.field_fc = nn.Sequential(
            nn.Linear(40, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # 6. Global Context Fusion: (P1 pooled + P2 pooled + Field) -> 512
        fusion_dim = d_model * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )

        # Value Head: outputs scalar expected outcome in [-1.0, 1.0]
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

        # Policy Head: outputs action logits (4 moves, 4 tera-moves, 5 switches, 1 default)
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_actions)
        )

    def load_compatible_state_dict(self, weights):
        """Import a legacy field encoder with zero weights on added public inputs.

        Original checkpoint bytes stay unchanged. No fabricated knowledge is added;
        only future training can learn to use these new neural features.
        """
        weights = dict(weights)
        old_move = weights.get("move_fc.0.weight")
        if old_move is not None and old_move.shape[1] == 32:
            weights["move_fc.0.weight"] = torch.nn.functional.pad(old_move,(0,MOVE_DIM-32))
        old = weights.get("field_fc.0.weight")
        if old is not None and old.shape[1] == 16:
            weights["field_fc.0.weight"] = torch.nn.functional.pad(old,(0,24))
        old_mon = weights.get("mon_projector.0.weight")
        expected = self.mon_projector[0].in_features
        if old_mon is not None and old_mon.shape[1] in (128+64,128+68,128+83):
            weights["mon_projector.0.weight"] = torch.nn.functional.pad(old_mon,(0,expected-old_mon.shape[1]))
            if old_mon.shape[1] == 128+64 and "field_fc.0.weight" in weights:
                weights["field_fc.0.weight"] = weights["field_fc.0.weight"].clone()
                weights["field_fc.0.weight"][:,37:40] = 0
        return self.load_state_dict(weights)

    def _encode_team(self, moves_t: torch.Tensor, stats_t: torch.Tensor) -> torch.Tensor:
        # moves_t: (batch, 6, 4, MOVE_DIM)
        # stats_t: (batch, 6, STAT_DIM)
        if moves_t.shape[-1] == 32:
            moves_t = torch.nn.functional.pad(moves_t,(0,MOVE_DIM-32))
        b, num_mons, num_moves, m_dim = moves_t.shape

        # Encode moves and pool across the 4 moves
        m_flat = moves_t.view(b * num_mons, num_moves, m_dim)
        m_emb = self.move_fc(m_flat)  # (b * 6, 4, move_emb_dim)
        m_pooled = torch.mean(m_emb, dim=1)  # (b * 6, move_emb_dim)
        m_pooled = m_pooled.view(b, num_mons, -1)

        # Concatenate move embeddings with rich Pokémon stat vector
        if stats_t.shape[-1] in (64,68,83):
            stats_t = torch.nn.functional.pad(stats_t,(0,STAT_DIM-stats_t.shape[-1]))
        mon_features = torch.cat([m_pooled, stats_t], dim=-1)  # (batch, 6, move_emb_dim + STAT_DIM)
        mon_tokens = self.mon_projector(mon_features)  # (batch, 6, d_model)

        # Permutation-Equivariant Team Set Attention
        h = self.team_sab1(mon_tokens)
        h = self.team_sab2(h)
        h = self.team_sab3(h)
        return h

    def forward(
        self,
        p1_moves: torch.Tensor,
        p1_stats: torch.Tensor,
        p2_moves: torch.Tensor,
        p2_stats: torch.Tensor,
        field: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass taking entire 6v6 state tensors.

        Returns:
            value: Scalar in [-1, 1] of shape (batch, 1)
            policy_logits: Action distribution logits of shape (batch, num_actions)
        """
        # Auto-batch if single unbatched item is passed
        if p1_moves.dim() == 3:
            p1_moves = p1_moves.unsqueeze(0)
            p1_stats = p1_stats.unsqueeze(0)
            p2_moves = p2_moves.unsqueeze(0)
            p2_stats = p2_stats.unsqueeze(0)
            field = field.unsqueeze(0)

        # 1. Encode both teams with Set Attention
        p1_tokens = self._encode_team(p1_moves, p1_stats)  # (batch, 6, d_model)
        p2_tokens = self._encode_team(p2_moves, p2_stats)  # (batch, 6, d_model)

        # 2. Bidirectional Cross-Attention (Matchup threat assessment)
        p1_attended = self.cross_attn1(p1_tokens, p2_tokens)  # (batch, 6, d_model)
        p2_attended = self.cross_attn2(p2_tokens, p1_tokens)  # (batch, 6, d_model)

        # 3. Permutation-Invariant Team Pooling (Mean + Max pooling)
        p1_pooled = torch.mean(p1_attended, dim=1)  # (batch, d_model)
        p2_pooled = torch.mean(p2_attended, dim=1)  # (batch, d_model)

        # 4. Field conditions
        if field.shape[-1] == 16:  # Explicit legacy input, missing public fields.
            field = torch.nn.functional.pad(field, (0,24))
        field_emb = self.field_fc(field)  # (batch, d_model)

        # 5. Global state fusion
        global_context = torch.cat([p1_pooled, p2_pooled, field_emb], dim=-1)  # (batch, d_model * 3)
        latent = self.fusion(global_context)  # (batch, 256)

        # 6. Heads
        value = self.value_head(latent)  # (batch, 1)
        policy_logits = self.policy_head(latent)  # (batch, num_actions)

        return value, policy_logits
