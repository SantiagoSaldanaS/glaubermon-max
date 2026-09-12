import asyncio,torch
from test_public_fields import bot
from test_live_observations import request
from glaubermon.models.embeddings import encode_battle_state,encode_pokemon
from glaubermon.models.set_transformer import GlaubermonMaxNet
from glaubermon.core.actions import MoveAction
from glaubermon.search.subgame_resolver import simulate_turn_transition


def test_public_move_identity_restrictions_and_seed_perspective():
 b=bot();room='battle-restrictions'
 asyncio.run(b.handle_message('>'+room+'\n|player|p1|GlaubermonAI|\n|poke|p2|Great Tusk, L100|\n|switch|p1a: Great Tusk|Great Tusk, L100|100/371\n|switch|p2a: Great Tusk|Great Tusk, L100|100/100\n|move|p2a: Great Tusk|Rapid Spin|p1a: Great Tusk\n|-start|p2a: Great Tusk|Encore\n|-start|p2a: Great Tusk|Disable|Rapid Spin\n|-start|p2a: Great Tusk|move: Leech Seed'))
 state=b.build_battle_state(room,request());mon=state.p2.active_pokemon
 assert mon.last_move=='rapidspin'
 assert mon.volatiles['encore']=={'duration':-1,'move':'rapidspin'}
 assert mon.volatiles['disable']=={'duration':-1,'move':'rapidspin'}
 assert mon.volatiles['leechseed']=={'source_side':1}
 tensors=encode_battle_state(state)
 assert torch.equal(tensors[1][1][0],encode_pokemon(mon,True)[1])
 assert tensors[1][1][0,68:71].tolist()==[1,1,1]
 slot=next(i for i,m in enumerate(mon.moves) if m.id=='rapidspin')
 assert all(tensors[1][1][0,offset+slot]==1 for offset in (71,75,79))
 assert state.flipped().p1.active_pokemon.volatiles['leechseed']['source_side']==2
 clone=state.clone();clone.p2.active_pokemon.volatiles['encore']['move']='splash'
 assert mon.volatiles['encore']['move']=='rapidspin'
 # Slot-bound healing persists when the seeder leaves; target-side switch clears.
 asyncio.run(b.handle_message('>'+room+'\n|switch|p1a: Replacement|Blissey, L100|100/100'))
 assert b.public_volatiles[room].observations('p2','Great Tusk',{'p1':(1,state.p1),'p2':(2,state.p2)})['leechseed']['source_side']==1
 asyncio.run(b.handle_message('>'+room+'\n|-end|p2a: Great Tusk|Encore\n|-end|p2a: Great Tusk|Disable\n|switch|p2a: Great Tusk|Great Tusk, L100|100/100'))
 after=b.build_battle_state(room,request());assert not after.p2.active_pokemon.volatiles
 assert after.p2.active_pokemon.last_move is None
 assert not b.build_battle_state('battle-unrelated',request()).p1.active_pokemon.volatiles


def test_request_disabled_preserves_pp_and_is_not_a_permanent_exhaustion():
 b=bot();req=request()
 req['active'][0]['moves']=[{'id':'seismictoss','pp':17,'maxpp':32,'disabled':True},{'id':'splash','pp':40,'maxpp':64,'disabled':False}]
 state=b.build_battle_state('battle-disabled',req)
 assert state.p1.active_pokemon.moves[0].pp==17
 assert 'seismictoss' not in [getattr(a,'move_id',None) for a in state.get_valid_actions(1)]
 state.p2.active_pokemon.moves=[state.p1.active_pokemon.moves[1].clone()]
 after=simulate_turn_transition(state,MoveAction('splash',2),MoveAction('splash',1))
 assert after.p1.active_pokemon.moves[0].pp==17
 assert 'seismictoss' in [getattr(a,'move_id',None) for a in after.get_valid_actions(1)]


def test_v4_checkpoint_zero_extends_only_new_columns():
 model=GlaubermonMaxNet(d_model=32,nhead=4).eval()
 weights={k:v.clone() for k,v in model.state_dict().items()}
 weights['mon_projector.0.weight']=weights['mon_projector.0.weight'][:,:196].clone()
 field=weights['field_fc.0.weight'].clone()
 model.load_compatible_state_dict(weights)
 assert torch.equal(model.field_fc[0].weight,field)
 assert weights['mon_projector.0.weight'].shape[1]==196
 assert torch.count_nonzero(model.mon_projector[0].weight[:,196:])==0
 token=torch.randn(2,model.mon_projector[0].in_features)
 baseline=token.clone();baseline[:,196:]=0
 assert torch.equal(model.mon_projector(token),model.mon_projector(baseline))

 # Explicit v4 inputs must use the same padding as v4 checkpoint weights.
 moves=torch.randn(1,6,4,32);stats=torch.randn(1,6,68);field=torch.randn(1,40)
 padded=torch.nn.functional.pad(stats,(0,15))
 with torch.no_grad():
  legacy=model(moves,stats,moves,stats,field)
  expanded=model(moves,padded,moves,padded,field)
 assert all(torch.equal(a,b) for a,b in zip(legacy,expanded))
