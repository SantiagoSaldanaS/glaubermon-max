"""Unit tests for the Permutation-Invariant Set Transformer network."""

import pytest
import torch
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory
from glaubermon.models.embeddings import encode_battle_state
from glaubermon.models.set_transformer import GlaubermonMaxNet


def make_dummy_state() -> BattleState:
    moves1 = [
        Move.create("Shadow Ball", PokemonType.GHOST, MoveCategory.SPECIAL, base_power=80),
        Move.create("Draco Meteor", PokemonType.DRAGON, MoveCategory.SPECIAL, base_power=130),
    ]
    p1_team = [
        Pokemon(species="Dragapult", types=(PokemonType.DRAGON, PokemonType.GHOST), moves=moves1),
        Pokemon(species="Great Tusk", types=(PokemonType.GROUND, PokemonType.FIGHTING)),
        Pokemon(species="Kingambit", types=(PokemonType.DARK, PokemonType.STEEL)),
        Pokemon(species="Iron Valiant", types=(PokemonType.FAIRY, PokemonType.FIGHTING)),
        Pokemon(species="Gholdengo", types=(PokemonType.STEEL, PokemonType.GHOST)),
        Pokemon(species="Ogerpon", types=(PokemonType.GRASS, None)),
    ]

    p2_team = [
        Pokemon(species="Dondozo", types=(PokemonType.WATER, None)),
        Pokemon(species="Clodsire", types=(PokemonType.POISON, PokemonType.GROUND)),
        Pokemon(species="Corviknight", types=(PokemonType.FLYING, PokemonType.STEEL)),
        Pokemon(species="Ting-Lu", types=(PokemonType.DARK, PokemonType.GROUND)),
        Pokemon(species="Garganacl", types=(PokemonType.ROCK, None)),
        Pokemon(species="Blissey", types=(PokemonType.NORMAL, None)),
    ]

    return BattleState(p1=BattleSide(pokemon=p1_team), p2=BattleSide(pokemon=p2_team))


def test_set_transformer_forward_pass():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GlaubermonMaxNet(d_model=128, nhead=4).to(device)
    net.eval()

    state = make_dummy_state()
    (p1_m, p1_s), (p2_m, p2_s), field = encode_battle_state(state)

    p1_m = p1_m.to(device)
    p1_s = p1_s.to(device)
    p2_m = p2_m.to(device)
    p2_s = p2_s.to(device)
    field = field.to(device)

    with torch.no_grad():
        value, policy = net(p1_m, p1_s, p2_m, p2_s, field)

    assert value.shape == (1, 1)
    assert -1.0 <= value.item() <= 1.0
    assert policy.shape == (1, 14)


def test_permutation_invariance():
    """Verify that permuting bench Pokémon positions does not alter the output value."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GlaubermonMaxNet(d_model=64, nhead=2).to(device)
    net.eval()

    state1 = make_dummy_state()
    (p1_m1, p1_s1), (p2_m1, p2_s1), field1 = encode_battle_state(state1)

    # Permute bench slots 2 and 4 in state2
    state2 = make_dummy_state()
    state2.p1.pokemon[2], state2.p1.pokemon[4] = state2.p1.pokemon[4], state2.p1.pokemon[2]
    (p1_m2, p1_s2), (p2_m2, p2_s2), field2 = encode_battle_state(state2)

    with torch.no_grad():
        val1, _ = net(p1_m1.to(device), p1_s1.to(device), p2_m1.to(device), p2_s1.to(device), field1.to(device))
        val2, _ = net(p1_m2.to(device), p1_s2.to(device), p2_m2.to(device), p2_s2.to(device), field2.to(device))

    assert torch.allclose(val1, val2, atol=1e-4), (
        f"Model is not permutation invariant! val1={val1.item():.5f}, val2={val2.item():.5f}"
    )
