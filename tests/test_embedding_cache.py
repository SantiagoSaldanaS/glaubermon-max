import torch
from glaubermon.models.embeddings import encode_move, encode_pokemon, encode_battle_state
from glaubermon.core.pokemon import Move, Pokemon
from glaubermon.core.types import PokemonType, MoveCategory
from glaubermon.core.battle_state import BattleState, BattleSide


def test_cache_distinguishes_dynamic_features_and_returns_independent_tensors():
    move = Move.from_dex('ivycudgel')
    first = encode_move(move)
    for attr, value in [('move_type',PokemonType.WATER), ('base_power',120),
                        ('accuracy',0.75), ('priority',2), ('pp',3), ('category',MoveCategory.SPECIAL)]:
        before = encode_move(move)
        setattr(move,attr,value)
        after = encode_move(move)
        assert not torch.equal(before,after), attr
    first.fill_(77)
    assert not torch.any(encode_move(Move.from_dex('ivycudgel')) == 77)


def test_preallocation_keeps_padding_fainted_and_stats_contract():
    alive = Pokemon(species='Snorlax', moves=[Move.from_dex('tackle')],
                    current_hp=100, max_hp=461, raw_stats={'hp':461,'spe':96})
    dead = alive.clone()
    dead.current_hp = 0
    state = BattleState(BattleSide([alive,dead]), BattleSide([alive.clone()], active_index=0))
    encoded = encode_battle_state(state)
    for side, tensors in zip((state.p1,state.p2),encoded[:2]):
        for i in range(6):
            mon = side.pokemon[i] if i < len(side.pokemon) else None
            expected = encode_pokemon(mon,is_active=i==side.active_index)
            assert torch.equal(tensors[0][i],expected[0])
            assert torch.equal(tensors[1][i],expected[1])
