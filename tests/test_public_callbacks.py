import asyncio
import torch
from test_public_fields import bot
from test_live_observations import request
from glaubermon.core.types import PokemonType
from glaubermon.models.embeddings import encode_battle_state,encode_pokemon,STAT_DIM
from glaubermon.models.set_transformer import GlaubermonMaxNet


def test_public_type_change_and_callback_state_survive_adapter_and_flip():
 b=bot();room='battle-callbacks'
 asyncio.run(b.handle_message('>'+room+'\n|player|p1|GlaubermonAI|\n|poke|p2|Meowscarada|\n|poke|p2|Heatran|\n|switch|p2a: Meowscarada|Meowscarada|100/100\n|-start|p2a: Meowscarada|typechange|Bug|[from] ability: Protean'))
 state=b.build_battle_state(room,request());m=state.p2.active_pokemon
 assert m.types==(PokemonType.GRASS,PokemonType.DARK)
 assert m.active_types==(PokemonType.BUG,None) and m.protean_used
 encoded=encode_battle_state(state)[1][1][state.p2.active_index]
 assert torch.equal(encoded,encode_pokemon(m,True)[1])
 assert encoded[83:86].tolist()==[1,0,0]
 assert state.flipped().p1.active_pokemon.active_types==m.active_types
 clone=state.clone();clone.p2.active_pokemon.type_override=None
 assert m.active_types==(PokemonType.BUG,None)
 asyncio.run(b.handle_message('>'+room+'\n|switch|p2a: Heatran|Heatran|100/100\n|-start|p2a: Heatran|ability: Flash Fire'))
 second=b.build_battle_state(room,request());heatran=second.p2.active_pokemon
 assert encode_pokemon(heatran,True)[1][85]==1
 asyncio.run(b.handle_message('>'+room+'\n|-end|p2a: Heatran|ability: Flash Fire|[silent]\n|switch|p2a: Meowscarada|Meowscarada|100/100'))
 final=b.build_battle_state(room,request()).p2.active_pokemon
 assert final.type_override is None and not final.protean_used
 assert final.active_types==final.types
 assert not b.build_battle_state('battle-unrelated',request()).p1.active_pokemon.volatiles


def test_public_roost_is_temporary_and_tera_keeps_type():
 b=bot();room='battle-roost'
 asyncio.run(b.handle_message('>'+room+'\n|poke|p2|Corviknight|\n|switch|p2a: Corviknight|Corviknight|100/100\n|-singleturn|p2a: Corviknight|move: Roost'))
 m=b.build_battle_state(room,request()).p2.active_pokemon
 assert m.active_types==(PokemonType.STEEL,None)
 assert encode_pokemon(m)[1][84]==1
 m.is_terastallized=True;m.tera_type=PokemonType.FLYING
 assert m.active_types==(PokemonType.FLYING,None)
 asyncio.run(b.handle_message('>'+room+'\n|upkeep\n|turn|2'))
 m=b.build_battle_state(room,request()).p2.active_pokemon
 assert PokemonType.FLYING in m.active_types and 'roost' not in m.volatiles


def test_v6_checkpoint_and_inputs_zero_extend_callbacks_only():
 model=GlaubermonMaxNet(d_model=32,nhead=4).eval()
 weights={k:v.clone() for k,v in model.state_dict().items()}
 weights['mon_projector.0.weight']=weights['mon_projector.0.weight'][:,:128+83].clone()
 model.load_compatible_state_dict(weights)
 assert torch.count_nonzero(model.mon_projector[0].weight[:,128+83:])==0
 assert weights['mon_projector.0.weight'].shape[1]==128+83
 moves=torch.randn(1,6,4,33);stats=torch.randn(1,6,83);field=torch.randn(1,40)
 padded=torch.nn.functional.pad(stats,(0,STAT_DIM-83))
 with torch.no_grad():
  old=model(moves,stats,moves,stats,field)
  new=model(moves,padded,moves,padded,field)
 assert all(torch.equal(a,b) for a,b in zip(old,new))


def test_wellspring_public_tera_counts_only_authoritative_boost():
 b=bot();room='battle-wellspring'
 asyncio.run(b.handle_message('>'+room+'\n|player|p1|GlaubermonAI|\n|poke|p2|Ogerpon-Wellspring|\n|switch|p2a: Ogerpon|Ogerpon-Wellspring|100/100\n|-terastallize|p2a: Ogerpon|Water\n|-boost|p2a: Ogerpon|spd|1|[from] ability: Embody Aspect'))
 m=b.build_battle_state(room,request()).p2.active_pokemon
 assert m.boosts['spd']==1 and m.ability=='embodyaspectwellspring'
 assert m.is_terastallized and m.active_types==(PokemonType.WATER,None)
 asyncio.run(b.handle_message('>'+room+'\n|switch|p2a: Ogerpon|Ogerpon-Wellspring-Tera|100/100\n|-boost|p2a: Ogerpon|spd|1|[from] ability: Embody Aspect'))
 m=b.build_battle_state(room,request()).p2.active_pokemon
 assert m.boosts['spd']==1 and m.ability=='embodyaspectwellspring'


def test_forced_replacement_does_not_consume_future_tera():
 from glaubermon.core.actions import SwitchAction
 from glaubermon.search.subgame_resolver import simulate_turn_transition
 b=bot();room='battle-replacement-tera';req=request()
 req.pop('active');req['forceSwitch']=[True]
 req['side']['pokemon'][0]['condition']='0 fnt'
 bench=dict(req['side']['pokemon'][0]);bench.update(ident='p1: Blissey',details='Blissey',active=False,condition='100/651',moves=['splash'],ability='naturalcure',teraType='Water')
 req['side']['pokemon'].append(bench)
 state=b.build_battle_state(room,req)
 assert not state.p1.is_tera_used
 after=simulate_turn_transition(state,SwitchAction(2,'Blissey'),None)
 assert any(getattr(a,'is_tera',False) for a in after.get_valid_actions(1))
 b.our_tera_used[room]=True
 assert b.build_battle_state(room,req).p1.is_tera_used
