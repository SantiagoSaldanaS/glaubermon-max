"""The same observation builder is used by live play and official self-play."""
import asyncio
import torch
from glaubermon.client.showdown_bot import ShowdownBot
from glaubermon.models.set_transformer import GlaubermonMaxNet
from glaubermon.models.embeddings import encode_battle_state


def observer():
    return ShowdownBot(model=GlaubermonMaxNet(d_model=32,nhead=4),load_config=False,stealth=False)


def request():
    return {'side': {'id':'p1', 'pokemon': [
        {'ident':'p1: Great Tusk','details':'Great Tusk, L100','active':True,
         'condition':'100/371','stats':{'atk':361,'def':298,'spa':127,'spd':143,'spe':300},
         'moves':['closecombat','headlongrush','icespinner','rapidspin'],
         'item':'', 'ability':'protosynthesis','teraType':'Ice'}]},
        'active':[{'moves':[{'id':'struggle','pp':1,'maxpp':1,'disabled':False}]}]}


def test_short_official_move_list_and_public_turn_are_preserved():
    bot=observer()
    asyncio.run(bot.handle_message('>battle-test\n|turn|47\n|poke|p2|Gholdengo, L100|'))
    state=bot.build_battle_state('battle-test',request())
    assert [m.id for m in state.p1.active_pokemon.moves] == request()['side']['pokemon'][0]['moves']
    assert all(not m.pp_known for m in state.p1.active_pokemon.moves)
    assert [a.move_id for a in state.get_valid_actions(1)] == ['struggle']
    assert state.turn == 47
    assert state.p1.active_pokemon.item is None


def test_hidden_opponent_sets_produce_identical_observations():
    from pathlib import Path
    import pytest
    from glaubermon.client.showdown_bot import PACKED_TEAMS
    from glaubermon.evaluation.official_benchmark import OfficialBridge
    showdown=Path(__file__).resolve().parents[2]/'showdown-parity/node_modules/pokemon-showdown'
    if not showdown.exists():
        pytest.skip('Official simulator unavailable')
    original=PACKED_TEAMS['balance']
    modified=original.replace('Gholdengo||airballoon','Gholdengo||choicespecs').replace(
        'makeitrain,shadowball,nastyplot,recover|Timid|,,,252,4,252',
        'makeitrain,shadowball,nastyplot,thunderbolt|Modest|,,,4,252,252')
    bridge=OfficialBridge(showdown)
    states=[]
    try:
        for opponent in (original,modified):
            bot=observer()
            frame=bridge.exchange(dict(op='start',teams=[original,opponent],
                                       names=['GlaubermonAI','Other'],seed=[1,2,3,4]))
            asyncio.run(bot.handle_message('>battle-test\n'+'\n'.join(frame['players'][0]['lines'])))
            frame=bridge.exchange(dict(op='choose',choices=['team 123456','team 123456']))
            channel=frame['players'][0]
            asyncio.run(bot.handle_message('>battle-test\n'+'\n'.join(channel['lines'])))
            states.append(encode_battle_state(bot.build_battle_state('battle-test',channel['request'])))
        for side in (0,1):
            assert all(torch.equal(a,b) for a,b in zip(states[0][side],states[1][side]))
    finally:
        bridge.close()


def test_revealed_move_changes_opponent_observation():
    bot=observer()
    asyncio.run(bot.handle_message('>battle-test\n|poke|p2|Gholdengo, L100|\n|switch|p2a: Gholdengo|Gholdengo, L100|100/100'))
    before=bot.build_battle_state('battle-test',request()).p2.active_pokemon.moves
    asyncio.run(bot.handle_message('>battle-test\n|move|p2a: Gholdengo|Thunderbolt|p1a: Great Tusk'))
    after=bot.build_battle_state('battle-test',request()).p2.active_pokemon.moves
    assert before[0].id != 'thunderbolt' and after[0].id == 'thunderbolt'
