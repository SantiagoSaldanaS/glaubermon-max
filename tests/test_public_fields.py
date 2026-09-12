import asyncio,torch
from glaubermon.client.showdown_bot import ShowdownBot
from glaubermon.models.set_transformer import GlaubermonMaxNet
from glaubermon.models.embeddings import encode_battle_state
from glaubermon.core.types import Terrain,StatusCondition
from test_live_observations import request


def bot():
 return ShowdownBot(model=GlaubermonMaxNet(d_model=32,nhead=4),load_config=False,stealth=False)


def test_fields_reach_both_search_and_network_without_guessing_hidden_clay():
 b=bot();room='battle-fields'
 asyncio.run(b.handle_message('>'+room+'\n|turn|3\n|-fieldstart|move: Grassy Terrain\n|-fieldstart|move: Trick Room\n|-sidestart|p2: Other|Reflect\n|-sidestart|p1: Us|Tailwind\n|turn|4'))
 state=b.build_battle_state(room,request())
 assert state.terrain==Terrain.GRASSY and state.trick_room==4
 assert state.p1.tailwind==3 and state.p2.screens=={'reflect':-1}
 field=encode_battle_state(state)[2]
 assert field.shape==(40,) and field[16]>0 and field[22]>0 and field[28]<0
 asyncio.run(b.handle_message('>'+room+'\n|-sideend|p2: Other|Reflect\n|-fieldend|move: Trick Room'))
 state=b.build_battle_state(room,request())
 assert not state.trick_room and not state.p2.screens and state.terrain==Terrain.GRASSY


def test_mirror_status_ability_item_and_rooms_are_isolated():
 b=bot();room='battle-mirror'
 asyncio.run(b.handle_message('>'+room+'\n|player|p1|GlaubermonAI|\n|poke|p2|Great Tusk, L100|\n|switch|p1a: Great Tusk|Great Tusk, L100|100/371\n|switch|p2a: Great Tusk|Great Tusk, L100|100/100\n|-status|p1a: Great Tusk|brn\n|-ability|p1a: Great Tusk|Levitate\n|-item|p1a: Great Tusk|Choice Band'))
 req=request();req['side']['pokemon'][0]['condition']='100/371 brn'
 state=b.build_battle_state(room,req)
 assert state.p1.active_pokemon.status==StatusCondition.BURN
 assert state.p2.active_pokemon.status==StatusCondition.NONE
 assert state.p2.active_pokemon.ability.lower()!='levitate'
 asyncio.run(b.handle_message('>'+room+'\n|-item|p2a: Great Tusk|Choice Scarf\n|-ability|p2a: Great Tusk|Levitate'))
 state=b.build_battle_state(room,req)
 assert state.p2.active_pokemon.item=='choice scarf'
 assert state.p2.active_pokemon.ability=='levitate'
 asyncio.run(b.handle_message('>battle-next\n|poke|p2|Great Tusk, L100|'))
 next_state=b.build_battle_state('battle-next',request())
 assert next_state.p2.active_pokemon.ability.lower()!='levitate'


def test_legacy_import_zero_extends_weights_and_preserves_outputs():
 model=GlaubermonMaxNet(d_model=32,nhead=4).eval()
 old=model.state_dict();old['field_fc.0.weight']=old['field_fc.0.weight'][:,:16].clone()
 model.load_compatible_state_dict(old)
 assert torch.count_nonzero(model.field_fc[0].weight[:,16:])==0
 fields=torch.rand(2,40)
 assert torch.equal(model.field_fc(fields),model.field_fc(torch.nn.functional.pad(fields[:,:16],(0,24))))


def test_training_stays_paused_before_rollouts_or_updates():
 import pytest
 from glaubermon.scripts.train_rebel import AlphaZeroTrainer
 trainer=object.__new__(AlphaZeroTrainer)
 with pytest.raises(RuntimeError,match='Training paused'):
  trainer.train(games_to_play=1)


def test_mirror_booster_effects_are_side_specific():
 b=bot();room='battle-mirror-booster'
 asyncio.run(b.handle_message('>'+room+'\n|player|p1|GlaubermonAI|\n|poke|p2|Great Tusk, L100|\n|switch|p1a: Great Tusk|Great Tusk, L100|100/371\n|switch|p2a: Great Tusk|Great Tusk, L100|100/100\n|-start|p1a: Great Tusk|protosynthesisatk'))
 state=b.build_battle_state(room,request())
 assert state.p1.active_pokemon.booster_stat=='atk'
 assert state.p2.active_pokemon.booster_stat is None
 asyncio.run(b.handle_message('>'+room+'\n|-start|p2a: Great Tusk|protosynthesisspe\n|-end|p1a: Great Tusk|Protosynthesis'))
 state=b.build_battle_state(room,request())
 assert state.p1.active_pokemon.booster_stat is None
 assert state.p2.active_pokemon.booster_stat=='spe'


def test_public_volatiles_are_side_specific_and_keep_hidden_values_unknown():
 from glaubermon.models.embeddings import encode_pokemon
 b=bot();room='battle-volatiles'
 asyncio.run(b.handle_message('>'+room+'\n|player|p1|GlaubermonAI|\n|poke|p2|Great Tusk, L100|\n|switch|p1a: Great Tusk|Great Tusk, L100|100/371\n|switch|p2a: Great Tusk|Great Tusk, L100|100/100\n|-start|p2a: Great Tusk|Substitute\n|-start|p2a: Great Tusk|move: Taunt\n|-start|p1a: Great Tusk|confusion\n|-activate|p1a: Great Tusk|move: Magma Storm|[of] p2a: Great Tusk'))
 state=b.build_battle_state(room,request())
 assert state.p1.active_pokemon.volatiles['confusion']=={'time':-1}
 assert state.p1.active_pokemon.volatiles['partiallytrapped']['source']==(2,0)
 assert state.p2.active_pokemon.volatiles['substitute']=={'hp':-1}
 assert state.p2.active_pokemon.volatiles['taunt']=={'duration':-1}
 assert 'substitute' not in state.p1.active_pokemon.volatiles
 assert state.is_trapped(1)
 tensors=encode_battle_state(state)
 assert tensors[0][1].shape==(6,86)
 assert tensors[1][1][0,64]==-1 and tensors[1][1][0,65]<0
 assert torch.equal(tensors[1][1][0],encode_pokemon(state.p2.active_pokemon,True)[1])
 asyncio.run(b.handle_message('>'+room+'\n|-end|p1a: Great Tusk|confusion\n|switch|p2a: Great Tusk|Great Tusk, L100|100/100'))
 state=b.build_battle_state(room,request())
 assert not state.p1.active_pokemon.volatiles and not state.p2.active_pokemon.volatiles


def test_public_force_request_reaches_actions_and_phase_features():
 from glaubermon.core.types import ActionType
 b=bot();req=request();req['forceSwitch']=[True]
 state=b.build_battle_state('battle-force-request',req)
 assert state.pending_switches==(1,)
 assert all(a.action_type==ActionType.SWITCH for a in state.get_valid_actions(1))
 assert state.get_valid_actions(2)==[]
 assert encode_battle_state(state)[2][37]==1


def test_legacy_token_import_zeros_new_observations_without_mutating_source():
 model=GlaubermonMaxNet(d_model=32,nhead=4).eval()
 old={key:value.clone() for key,value in model.state_dict().items()}
 old['mon_projector.0.weight']=old['mon_projector.0.weight'][:,:192].clone()
 original=old['field_fc.0.weight'].clone()
 model.load_compatible_state_dict(old)
 assert torch.equal(old['field_fc.0.weight'],original)
 assert torch.count_nonzero(model.mon_projector[0].weight[:,192:])==0
 assert torch.count_nonzero(model.field_fc[0].weight[:,37:40])==0
 token=torch.randn(2,model.mon_projector[0].in_features)
 baseline=token.clone();baseline[:,192:]=0
 assert torch.allclose(model.mon_projector(token),model.mon_projector(baseline),atol=1e-6)
