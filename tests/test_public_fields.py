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
