import asyncio
from copy import deepcopy
import pytest,torch
from test_live_observations import observer,request
from test_alignment_reference import SHOWDOWN
from glaubermon.client.own_move_history import OwnMoveHistory
from glaubermon.data.showdown_dex import ShowdownDex
from glaubermon.core.actions import MoveAction
from glaubermon.search.subgame_resolver import simulate_turn_transition
from glaubermon.models.embeddings import encode_move
from glaubermon.models.set_transformer import GlaubermonMaxNet


def test_official_struggle_keeps_moves_pp_and_recovers_after_encore():
 from glaubermon.evaluation.showdown_transport import OfficialBridge
 if not SHOWDOWN.exists():pytest.skip('Install pinned Showdown')
 b=observer();bridge=OfficialBridge(str(SHOWDOWN));room='battle-history'
 teams=['Slowbro||leftovers|oblivious|calmmind,surf,slackoff,protect|Bold|252,,252,,4,|||||,,,,,Water',
        'Gengar||leftovers|cursedbody|encore,disable,shadowball,protect|Timid|,,,252,4,252|||||,,,,,Ghost']
 def observe(frame):
  assert not frame['errors'],frame['errors']
  channel=frame['players'][0]
  asyncio.run(b.handle_message('>'+room+'\n'+'\n'.join(l for l in channel['lines'] if not l.startswith(('|request|','|win|','|tie|')))))
  return b.build_battle_state(room,channel['request'])
 try:
  frame=bridge.exchange(dict(op='start',teams=teams,names=['GlaubermonAI','Other'],seed=[81,2,3,4]));observe(frame)
  state=observe(bridge.exchange(dict(op='choose',choices=['team 1','team 1'])))
  initial_pp=state.p1.active_pokemon.moves[0].pp
  state=observe(bridge.exchange(dict(op='choose',choices=['move 1','move 4'])))
  state=observe(bridge.exchange(dict(op='choose',choices=['move 2','move 1'])))
  frame=bridge.exchange(dict(op='choose',choices=['move 1','move 2']));state=observe(frame)
  assert frame['players'][0]['request']['active'][0]['moves'][0]['id']=='struggle'
  assert [m.id for m in state.p1.active_pokemon.moves]==['calmmind','surf','slackoff','protect']
  assert state.p1.active_pokemon.moves[0].pp==initial_pp-2
  assert all(m.pp_known for m in state.p1.active_pokemon.moves)
  assert state.p1.active_pokemon.volatiles['encore']['duration']==1
  assert [getattr(a,'move_id',None) for a in state.get_valid_actions(1)]==['struggle']
  enemy=state.p2.active_pokemon
  protect_slot=next(i+1 for i,m in enumerate(enemy.moves) if m.id=='protect')
  child=simulate_turn_transition(state,MoveAction('struggle',1),MoveAction('protect',protect_slot))
  assert 'surf' in [getattr(a,'move_id',None) for a in child.get_valid_actions(1)]
  assert child.p1.active_pokemon.moves[0].pp==initial_pp-2
  child.p1.active_pokemon.moves[0].pp=0
  assert state.p1.active_pokemon.moves[0].pp==initial_pp-2
  frame=bridge.exchange(dict(op='choose',choices=['move 1','move 4']));state=observe(frame)
  assert 'surf' in [getattr(a,'move_id',None) for a in state.get_valid_actions(1)]
  assert state.p1.active_pokemon.moves[0].pp==initial_pp-2
  # Repeated observation must neither spend nor restore PP.
  again=b.build_battle_state(room,frame['players'][0]['request'])
  assert [m.pp for m in again.p1.active_pokemon.moves]==[m.pp for m in state.p1.active_pokemon.moves]
 finally:bridge.close()


def test_pp_expenditure_pressure_called_moves_and_bench():
 h=OwnMoveHistory(ShowdownDex.get_instance());p=request()['side']['pokemon'][0]
 h.ingest('|start'.split('|'));h.ingest('|switch|p2a: Corviknight|Corviknight|100/100'.split('|'))
 h.ingest('|-ability|p2a: Corviknight|Pressure'.split('|'))
 menu=[{'id':'closecombat','pp':3,'maxpp':8}]
 h.observe(p,menu)
 h.ingest('|move|p1a: Great Tusk|Close Combat|p2a: Corviknight'.split('|'))
 moves=h.observe({**p,'moves':['closecombat']})
 assert moves[0].pp==1 and moves[0].pp_known
 h.ingest('|move|p1a: Great Tusk|Close Combat|p2a: Corviknight|[from] move: Sleep Talk'.split('|'))
 assert h.observe({**p,'moves':['closecombat']})[0].pp==1
 h.ingest('|move|p1a: Great Tusk|Close Combat|p2a: Corviknight'.split('|'))
 assert h.observe({**p,'moves':['closecombat']})[0].pp==0


def test_cold_struggle_is_unknown_until_a_real_request_and_rooms_are_isolated():
 b=observer();cold=b.build_battle_state('cold',request())
 assert cold.p1.active_pokemon.move_history_incomplete
 assert all(not m.pp_known for m in cold.p1.active_pokemon.moves)
 req=deepcopy(request());req['active'][0]['moves']=[{'id':'closecombat','pp':6,'maxpp':8}]
 known=b.build_battle_state('cold',req)
 assert not known.p1.active_pokemon.move_history_incomplete
 assert known.p1.active_pokemon.moves[0].pp==6
 other=b.build_battle_state('other',request())
 assert all(not m.pp_known for m in other.p1.active_pokemon.moves)


def test_legacy_move_encoder_ignores_added_confidence_flag():
 model=GlaubermonMaxNet(d_model=32,nhead=4).eval()
 weights={k:v.clone() for k,v in model.state_dict().items()}
 weights['move_fc.0.weight']=weights['move_fc.0.weight'][:,:32].clone()
 model.load_compatible_state_dict(weights)
 move=ShowdownDex.get_instance().get_move('surf');known=encode_move(move)
 move.pp_known=False;unknown=encode_move(move)
 assert known.shape==(33,) and known[32]==1 and unknown[32]==0
 assert torch.equal(known[:32],unknown[:32])
 assert torch.equal(model.move_fc(known),model.move_fc(unknown))
 assert torch.count_nonzero(model.move_fc[0].weight[:,32:])==0
 assert weights['move_fc.0.weight'].shape[1]==32
