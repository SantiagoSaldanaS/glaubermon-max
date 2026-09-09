"""Unit tests for exact Nash Equilibrium matrix game solving."""

import numpy as np
import pytest
from glaubermon.search.matrix_solver import solve_zero_sum_game, regret_matching_solve


def test_rock_paper_scissors_lp():
    # R P S: Payoffs are 1 for win, -1 for loss, 0 for tie
    # Row: P1, Col: P2
    M = np.array([
        [0.0, -1.0, 1.0],   # Rock vs (Rock, Paper, Scissors)
        [1.0, 0.0, -1.0],   # Paper vs ...
        [-1.0, 1.0, 0.0]    # Scissors vs ...
    ])

    p1_strat, p2_strat, game_val = solve_zero_sum_game(M)

    expected = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert np.allclose(p1_strat, expected, atol=1e-3), f"P1 strat {p1_strat} != 1/3 each"
    assert np.allclose(p2_strat, expected, atol=1e-3), f"P2 strat {p2_strat} != 1/3 each"
    assert np.isclose(game_val, 0.0, atol=1e-3), f"Game value {game_val} != 0.0"


def test_matching_pennies():
    # P1 wants to match (1), P2 wants to differ (-1)
    M = np.array([
        [1.0, -1.0],
        [-1.0, 1.0]
    ])

    p1_strat, p2_strat, game_val = solve_zero_sum_game(M)
    assert np.allclose(p1_strat, [0.5, 0.5], atol=1e-3)
    assert np.allclose(p2_strat, [0.5, 0.5], atol=1e-3)
    assert np.isclose(game_val, 0.0, atol=1e-3)


def test_asymmetric_matrix():
    # Asymmetric payoff matrix where one strategy dominates
    M = np.array([
        [3.0, 1.0],
        [0.0, 2.0]
    ])
    p1_strat, p2_strat, game_val = solve_zero_sum_game(M)
    assert len(p1_strat) == 2
    assert len(p2_strat) == 2
    assert np.isclose(np.sum(p1_strat), 1.0)
    assert np.isclose(np.sum(p2_strat), 1.0)
    assert game_val > 0.0


def test_regret_matching_plus():
    M = np.array([
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0]
    ])
    p1_strat, p2_strat, game_val = regret_matching_solve(M, iterations=2000)
    expected = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert np.allclose(p1_strat, expected, atol=0.05)
    assert np.allclose(p2_strat, expected, atol=0.05)


def test_trembling_hand_eliminates_weak_dominance():
    """Verify that Selten (1975) Trembling Hand Perfection selects the weakly dominating strategy
    when an opponent's lethal response creates a degenerate minimax worst-case plateau.
    
    In this game:
    Col 2 (e.g. Iron Head) KOs P1 regardless of P1's move, yielding -1.50 across all rows.
    However, Row 0 (e.g. Earthquake) yields +1.50 against Cols 0, 1, and 3, whereas Rows 1, 2, 3
    yield 0.0. A naive LP solver can pick any degenerate row. Trembling hand perfection MUST
    strictly select Row 0.
    """
    M = np.array([
        [1.5, 1.5, -1.5, 1.5],   # Row 0 (Earthquake): Wins against all cols except Col 2
        [0.0, 0.0, -1.5, 0.0],   # Row 1 (Ruination): Weakly dominated
        [0.0, 0.0, -1.5, 0.0],   # Row 2 (Stealth Rock): Weakly dominated
        [0.0, 0.0, -1.5, 0.0],   # Row 3 (Whirlwind): Weakly dominated
    ])

    p1_strat, p2_strat, game_val = solve_zero_sum_game(M)

    # Row 0 MUST be selected with 100% probability
    assert p1_strat[0] > 0.99, f"Expected Row 0 to dominate, got p1_strat={p1_strat}"
    assert np.isclose(game_val, -1.5, atol=1e-2)

