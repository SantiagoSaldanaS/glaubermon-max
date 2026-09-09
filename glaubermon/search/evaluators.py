"""Battle State Evaluators for Subgame Leaf Evaluation."""

from abc import ABC, abstractmethod
from typing import Optional
from glaubermon.core.battle_state import BattleState
from glaubermon.core.types import StatusCondition, Hazard


class StateEvaluator(ABC):
    """Abstract interface for evaluating a battle state."""

    @abstractmethod
    def evaluate(self, state: BattleState) -> float:
        """Evaluate state from Player 1's perspective.
        
        Returns:
            Score in range [-1.0, 1.0], where 1.0 is a guaranteed P1 win and -1.0 is a P1 loss.
        """
        pass


class HeuristicEvaluator(StateEvaluator):
    """Domain-expert competitive heuristic evaluation function."""

    def evaluate(self, state: BattleState) -> float:
        if state.is_game_over:
            return 1.0 if state.winner == 1 else -1.0

        p1 = state.p1
        p2 = state.p2

        # 1. Total HP fractions (scaled by dynamic role value and team size)
        def get_mon_scaling_factor(mon, fallen_count: int) -> float:
            if mon.is_fainted:
                return 0.0
            factor = 1.0
            ab = (mon.ability or "").lower().replace("-", "").replace(" ", "")
            if ab == "supremeoverlord":
                factor += 0.10 * min(5, fallen_count)
            return factor

        # Account for hazard-lethal bench Pokémon (a bench mon that dies to hazards on entry is functionally fainted)
        def is_mon_effectively_alive(mon, side) -> bool:
            if mon.is_fainted:
                return False
            if side.active_pokemon is mon:
                return True
            return not mon.is_dead_to_hazards(side.hazards)

        p1_fallen = sum(1 for p in p1.pokemon if not is_mon_effectively_alive(p, p1))
        p2_fallen = sum(1 for p in p2.pokemon if not is_mon_effectively_alive(p, p2))

        def mon_effective_hp_fraction(mon, side) -> float:
            if not is_mon_effectively_alive(mon, side):
                return 0.0
            active_mult = 1.5 if side.active_pokemon is mon else 1.0
            return mon.hp_percent * active_mult

        p1_hp_sum = sum(mon_effective_hp_fraction(p, p1) * get_mon_scaling_factor(p, p1_fallen) for p in p1.pokemon)
        p2_hp_sum = sum(mon_effective_hp_fraction(p, p2) * get_mon_scaling_factor(p, p2_fallen) for p in p2.pokemon)
        hp_score = max(-1.0, min(1.0, (p1_hp_sum - p2_hp_sum) / 6.5))

        # 2. Pokémon alive difference with smooth health degradation
        # A 5% HP mon is near faint; smooth weighting prevents perverse survival stalling over dealing heavy damage
        def mon_alive_value(mon, side) -> float:
            if not is_mon_effectively_alive(mon, side):
                return 0.0
            return min(1.0, 0.20 + 0.80 * mon.hp_percent)

        p1_alive_val = sum(mon_alive_value(p, p1) for p in p1.pokemon)
        p2_alive_val = sum(mon_alive_value(p, p2) for p in p2.pokemon)
        alive_diff = p1_alive_val - p2_alive_val
        alive_score = max(-1.0, min(1.0, alive_diff / 3.0))

        # 3. Hazard advantage
        p1_grounded = sum(1 for p in p1.pokemon if not p.is_fainted and p.is_grounded())
        p2_grounded = sum(1 for p in p2.pokemon if not p.is_fainted and p.is_grounded())

        hazard_score = 0.0
        if Hazard.STEALTH_ROCK in p2.hazards:
            hazard_score += 0.25
        if Hazard.STEALTH_ROCK in p1.hazards:
            hazard_score -= 0.25

        spikes_p2 = p2.hazards.get(Hazard.SPIKES_1, 0)
        spikes_p1 = p1.hazards.get(Hazard.SPIKES_1, 0)
        # Spikes deals 12.5%/16.7%/25% max HP per switch-in to grounded Pokémon
        hazard_score += (0.10 + 0.08 * p2_grounded) * spikes_p2
        hazard_score -= (0.10 + 0.08 * p1_grounded) * spikes_p1

        if Hazard.TOXIC_SPIKES_1 in p2.hazards:
            hazard_score += 0.10 * p2.hazards.get(Hazard.TOXIC_SPIKES_1, 0)
        if Hazard.TOXIC_SPIKES_1 in p1.hazards:
            hazard_score -= 0.10 * p1.hazards.get(Hazard.TOXIC_SPIKES_1, 0)

        if Hazard.STICKY_WEB in p2.hazards:
            hazard_score += 0.20
        if Hazard.STICKY_WEB in p1.hazards:
            hazard_score -= 0.20

        # 4. Status penalties
        status_penalty = 0.0
        for p in p1.pokemon:
            if not p.is_fainted:
                if p.status == StatusCondition.BURN and p.effective_stat("atk") > p.effective_stat("spa"):
                    status_penalty -= 0.08
                elif p.status in (StatusCondition.TOXIC, StatusCondition.FREEZE, StatusCondition.SLEEP):
                    status_penalty -= 0.06
        for p in p2.pokemon:
            if not p.is_fainted:
                if p.status == StatusCondition.BURN and p.effective_stat("atk") > p.effective_stat("spa"):
                    status_penalty += 0.08
                elif p.status in (StatusCondition.TOXIC, StatusCondition.FREEZE, StatusCondition.SLEEP):
                    status_penalty += 0.06

        # 5. Active Stat boosts
        boost_score = 0.0
        if p1.active_pokemon:
            boost_score += sum(p1.active_pokemon.boosts.values()) * 0.02
        if p2.active_pokemon:
            boost_score -= sum(p2.active_pokemon.boosts.values()) * 0.02

        # 6. Terastallization Resource Economics (Option Value)
        # Retaining Tera maintains strategic flexibility without distorting base leaf evaluations.
        tera_score = 0.0
        if not p1.is_tera_used:
            tera_score += 0.35 * min(1.0, p1_alive_val / 3.0)
        if not p2.is_tera_used:
            tera_score -= 0.35 * min(1.0, p2_alive_val / 3.0)

        # Weighted aggregate
        total_eval = (
            0.40 * hp_score +
            0.30 * alive_score +
            0.10 * hazard_score +
            0.10 * tera_score +
            0.06 * status_penalty +
            0.04 * boost_score
        )

        return max(-1.0, min(1.0, total_eval))

    def evaluate_batch(self, states: list):
        import numpy as np
        return np.array([self.evaluate(s) for s in states], dtype=np.float32)


class NeuralEvaluator(StateEvaluator):
    """Neural position evaluator and action policy prior provider."""

    def __init__(self, model, device=None):
        import torch
        self.model = model
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, state: BattleState) -> float:
        import torch
        from glaubermon.models.embeddings import encode_battle_state
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

    def evaluate_batch(self, states: list) -> "np.ndarray":
        import torch
        import numpy as np
        from glaubermon.models.embeddings import encode_battle_state
        if not states:
            return np.array([], dtype=np.float32)

        results = np.zeros(len(states), dtype=np.float32)
        valid_indices = []
        p1_m_list, p1_s_list, p2_m_list, p2_s_list, field_list = [], [], [], [], []

        for i, s in enumerate(states):
            if s.is_game_over:
                results[i] = 1.0 if s.winner == 1 else -1.0
            else:
                valid_indices.append(i)
                (p1_m, p1_s), (p2_m, p2_s), field = encode_battle_state(s)
                p1_m_list.append(p1_m)
                p1_s_list.append(p1_s)
                p2_m_list.append(p2_m)
                p2_s_list.append(p2_s)
                field_list.append(field)

        if valid_indices:
            batch_p1_m = torch.stack(p1_m_list).to(self.device)
            batch_p1_s = torch.stack(p1_s_list).to(self.device)
            batch_p2_m = torch.stack(p2_m_list).to(self.device)
            batch_p2_s = torch.stack(p2_s_list).to(self.device)
            batch_field = torch.stack(field_list).to(self.device)

            with torch.no_grad():
                vals, _ = self.model(batch_p1_m, batch_p1_s, batch_p2_m, batch_p2_s, batch_field)
            val_array = vals.squeeze(-1).cpu().numpy()
            for idx, v in zip(valid_indices, val_array):
                results[idx] = float(v)

        return results

    def get_policy_prior(self, state: BattleState, actions_or_count):
        import torch
        import numpy as np
        from glaubermon.models.embeddings import encode_battle_state
        from glaubermon.core.actions import Action, action_to_logit_index

        (p1_m, p1_s), (p2_m, p2_s), field = encode_battle_state(state)
        p1_m = p1_m.to(self.device)
        p1_s = p1_s.to(self.device)
        p2_m = p2_m.to(self.device)
        p2_s = p2_s.to(self.device)
        field = field.to(self.device)

        with torch.no_grad():
            _, logits = self.model(p1_m, p1_s, p2_m, p2_s, field)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        if isinstance(actions_or_count, list):
            actions = actions_or_count
            priors = np.zeros(len(actions), dtype=np.float32)
            for i, a in enumerate(actions):
                idx = action_to_logit_index(a)
                priors[i] = probs[idx]
            total = np.sum(priors)
            if total > 0:
                return priors / total
            return np.ones(len(actions)) / len(actions)
        else:
            num = int(actions_or_count)
            sliced = probs[:num]
            total = np.sum(sliced)
            if total > 0:
                return sliced / total
            return np.ones(num) / num


class HybridEvaluator(StateEvaluator):
    """Combines NeuralEvaluator's strategic foresight with HeuristicEvaluator's grounded material & KO value."""

    def __init__(
        self,
        neural_eval: Optional[StateEvaluator] = None,
        heuristic_eval: Optional[StateEvaluator] = None,
        weight_neural: float = 0.60,
        **kwargs
    ):
        self.neural = neural_eval or kwargs.get("neural_evaluator")
        self.heuristic = heuristic_eval or kwargs.get("heuristic_evaluator") or HeuristicEvaluator()
        self.w_n = weight_neural
        self.w_h = 1.0 - weight_neural

    def evaluate(self, state: BattleState) -> float:
        if state.is_game_over:
            return 1.0 if state.winner == 1 else -1.0
        nv = self.neural.evaluate(state)
        hv = self.heuristic.evaluate(state)
        return float(self.w_n * nv + self.w_h * hv)

    def evaluate_batch(self, states: list):
        import numpy as np
        if not states:
            return np.array([], dtype=np.float32)
        if hasattr(self.neural, "evaluate_batch"):
            n_vals = self.neural.evaluate_batch(states)
        elif self.neural is not None:
            n_vals = np.array([self.neural.evaluate(s) for s in states], dtype=np.float32)
        else:
            n_vals = np.zeros(len(states), dtype=np.float32)

        if hasattr(self.heuristic, "evaluate_batch"):
            h_vals = self.heuristic.evaluate_batch(states)
        elif self.heuristic is not None:
            h_vals = np.array([self.heuristic.evaluate(s) for s in states], dtype=np.float32)
        else:
            h_vals = np.zeros(len(states), dtype=np.float32)

        return self.w_n * n_vals + self.w_h * h_vals

    def get_policy_prior(self, state: BattleState, actions_or_count):
        return self.neural.get_policy_prior(state, actions_or_count)


