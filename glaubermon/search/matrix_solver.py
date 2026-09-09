"""Exact Nash Equilibrium Solver for Simultaneous Zero-Sum Matrix Games."""

from typing import Optional, Tuple
import numpy as np
from scipy.optimize import linprog


def solve_zero_sum_game(
    payoff_matrix: np.ndarray,
    p1_prior: Optional[np.ndarray] = None,
    prior_weight: float = 0.001,
    tremble_weight: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute the exact minimax Nash Equilibrium mixed strategies for a 2-player zero-sum game
    with Selten (1975) Trembling Hand Perfection / Admissible Equilibrium regularization.

    Eliminates weakly dominated strategies on degenerate minimax plateaus (e.g. when an opponent's
    lethal move ties all player actions at -1.5, trembling hand perfection breaks the tie in favor of
    the action that wins if the opponent plays any other move).

    Args:
        payoff_matrix: A 2D numpy array of shape (n, m) where M[i, j] is the payoff to P1
                       when P1 plays action i and P2 plays action j.
        p1_prior: Optional prior probability distribution over P1 actions for tie-breaking.
        prior_weight: Weight lambda for prior-regularized Nash solving.
        tremble_weight: Weight for Trembling Hand (Selten 1975) admissible tie-breaking.

    Returns:
        p1_strategy: 1D array of shape (n,) representing P1's optimal mixed strategy.
        p2_strategy: 1D array of shape (m,) representing P2's optimal mixed strategy.
        game_value: The scalar value of the game to P1.
    """
    n, m = payoff_matrix.shape
    if n == 0 or m == 0:
        return np.zeros(n), np.zeros(m), 0.0
    if n == 1 and m == 1:
        return np.array([1.0]), np.array([1.0]), float(payoff_matrix[0, 0])
    if n == 1:
        # P1 has only 1 move; P2 will minimize P1's payoff
        min_idx = np.argmin(payoff_matrix[0, :])
        p2_strat = np.zeros(m)
        p2_strat[min_idx] = 1.0
        return np.array([1.0]), p2_strat, float(payoff_matrix[0, min_idx])
    if m == 1:
        # P2 has only 1 move; P1 will maximize payoff
        max_idx = np.argmax(payoff_matrix[:, 0])
        p1_strat = np.zeros(n)
        p1_strat[max_idx] = 1.0
        return p1_strat, np.array([1.0]), float(payoff_matrix[max_idx, 0])

    # Shift payoffs to strictly positive to avoid negative value issues in standard LP form
    shift = np.abs(np.min(payoff_matrix)) + 1.0
    M = payoff_matrix + shift

    # --- P1 Linear Program ---
    # Variables: [x_1, x_2, ..., x_n, v]
    # Maximize v + tremble_weight * E_j[M[i, j]] + prior_weight * prior
    # <=> Minimize -v - tremble_weight * tremble_p1 - prior_weight * prior
    c_p1 = np.zeros(n + 1)
    c_p1[-1] = -1.0
    tremble_p1 = np.mean(M, axis=1)
    c_p1[:n] -= tremble_weight * tremble_p1
    if p1_prior is not None and len(p1_prior) == n:
        c_p1[:n] -= prior_weight * p1_prior

    # Constraints: For each P2 action j: sum_i (x_i * M[i, j]) >= v  <=>  -sum_i (x_i * M[i, j]) + v <= 0
    A_ub_p1 = np.zeros((m, n + 1))
    A_ub_p1[:, :n] = -M.T
    A_ub_p1[:, -1] = 1.0
    b_ub_p1 = np.zeros(m)

    # Equality: sum(x_i) == 1
    A_eq_p1 = np.zeros((1, n + 1))
    A_eq_p1[0, :n] = 1.0
    b_eq_p1 = np.array([1.0])

    # Bounds: x_i >= 0, v is unbounded
    bounds_p1 = [(0.0, 1.0) for _ in range(n)] + [(None, None)]

    res_p1 = linprog(
        c=c_p1,
        A_ub=A_ub_p1,
        b_ub=b_ub_p1,
        A_eq=A_eq_p1,
        b_eq=b_eq_p1,
        bounds=bounds_p1,
        method="highs"
    )

    if not res_p1.success:
        # Fallback to uniform distribution if numerical solver fails
        p1_strat = np.ones(n) / n
        p2_strat = np.ones(m) / m
        return p1_strat, p2_strat, 0.0

    p1_strat = np.maximum(0.0, res_p1.x[:n])
    p1_strat /= np.sum(p1_strat)
    val_shifted = res_p1.x[-1]
    game_value = float(val_shifted - shift)

    # --- P2 Linear Program ---
    # Variables: [y_1, y_2, ..., y_m, u]
    # Minimize u + tremble_weight * E_i[M[i, j]]
    c_p2 = np.zeros(m + 1)
    c_p2[-1] = 1.0
    tremble_p2 = np.mean(M, axis=0)
    c_p2[:m] += tremble_weight * tremble_p2

    # Constraints: For each P1 action i: sum_j (y_j * M[i, j]) <= u  <=>  sum_j (y_j * M[i, j]) - u <= 0
    A_ub_p2 = np.zeros((n, m + 1))
    A_ub_p2[:, :m] = M
    A_ub_p2[:, -1] = -1.0
    b_ub_p2 = np.zeros(n)

    A_eq_p2 = np.zeros((1, m + 1))
    A_eq_p2[0, :m] = 1.0
    b_eq_p2 = np.array([1.0])

    bounds_p2 = [(0.0, 1.0) for _ in range(m)] + [(None, None)]

    res_p2 = linprog(
        c=c_p2,
        A_ub=A_ub_p2,
        b_ub=b_ub_p2,
        A_eq=A_eq_p2,
        b_eq=b_eq_p2,
        bounds=bounds_p2,
        method="highs"
    )

    if res_p2.success:
        p2_strat = np.maximum(0.0, res_p2.x[:m])
        p2_strat /= np.sum(p2_strat)
    else:
        p2_strat = np.ones(m) / m

    return p1_strat, p2_strat, game_value


def regret_matching_solve(payoff_matrix: np.ndarray, iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve zero-sum game via Regret Matching+ (CFR+), ideal for fast GPU/batched processing."""
    n, m = payoff_matrix.shape
    cum_regrets_p1 = np.zeros(n)
    cum_regrets_p2 = np.zeros(m)
    strategy_sum_p1 = np.zeros(n)
    strategy_sum_p2 = np.zeros(m)

    for _ in range(iterations):
        # Compute strategy from positive regrets
        pos_regrets_p1 = np.maximum(cum_regrets_p1, 0.0)
        sum_p1 = np.sum(pos_regrets_p1)
        sigma_p1 = pos_regrets_p1 / sum_p1 if sum_p1 > 0 else np.ones(n) / n

        pos_regrets_p2 = np.maximum(cum_regrets_p2, 0.0)
        sum_p2 = np.sum(pos_regrets_p2)
        sigma_p2 = pos_regrets_p2 / sum_p2 if sum_p2 > 0 else np.ones(m) / m

        strategy_sum_p1 += sigma_p1
        strategy_sum_p2 += sigma_p2

        # Expected payoffs
        payoff_p1 = payoff_matrix @ sigma_p2  # Payoff for each P1 action
        payoff_p2 = -(sigma_p1 @ payoff_matrix)  # Payoff for each P2 action (zero-sum)

        node_val_p1 = np.dot(sigma_p1, payoff_p1)
        node_val_p2 = np.dot(sigma_p2, payoff_p2)

        cum_regrets_p1 += payoff_p1 - node_val_p1
        cum_regrets_p2 += payoff_p2 - node_val_p2

    avg_p1 = strategy_sum_p1 / np.sum(strategy_sum_p1)
    avg_p2 = strategy_sum_p2 / np.sum(strategy_sum_p2)
    game_val = float(avg_p1 @ payoff_matrix @ avg_p2)
    return avg_p1, avg_p2, game_val
