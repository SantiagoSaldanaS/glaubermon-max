"""Volatile conditions and paused replacement phases against pinned Showdown."""
import json, random, subprocess
from copy import deepcopy
import pytest
from test_alignment_reference import ROOT,SHOWDOWN,from_snapshot,STATUS
from glaubermon.core.actions import MoveAction,SwitchAction
from glaubermon.core.types import ActionType
from glaubermon.search.subgame_resolver import simulate_turn_transition


def team(species='Snorlax',moves=None,**kwargs):
 return dict(species=species,moves=moves or ['Splash'],**kwargs)

CASES=[
 dict(name='substitute_cost',moves=[['Substitute'],['Splash']]),
 dict(name='substitute_repeated_fails',moves=[['Substitute'],['Splash']],actions=[['move 1','move 1']]*2),
 dict(name='substitute_insufficient_hp',moves=[['Substitute'],['Splash']],initial=[{'hp':100},{}]),
 dict(name='substitute_absorbs_and_no_spill',moves=[['Splash'],['Seismic Toss']],initial=[{'volatiles':{'substitute':{'hp':115}}},{}],actions=[['move 1','move 1']]*2),
 dict(name='substitute_blocks_toxic',moves=[['Splash'],['Toxic']],initial=[{'volatiles':{'substitute':{'hp':115}}},{}]),
 dict(name='taunt_bypasses_substitute',teams=[[team(moves=['Taunt'],evs={'spe':252})],[team()]],initial=[{}, {'volatiles':{'substitute':{'hp':115}}}]),
 dict(name='fast_taunt_cancels_pending_status_without_pp',teams=[[team(moves=['Taunt'],evs={'spe':252})],[team(moves=['Swords Dance','Seismic Toss'])]],actions=[['move 1','move 1'],['move 1','move 2']]),
 dict(name='late_taunt_lasts_three_future_turns',teams=[[team(moves=['Taunt','Splash'])],[team(moves=['Splash','Seismic Toss'],evs={'spe':252})]],actions=[['move 1','move 1']]+[['move 2','move 2']]*3),
 dict(name='mental_herb_removes_taunt',teams=[[team(moves=['Taunt'],evs={'spe':252})],[team(item='Mental Herb')]]),
 dict(name='oblivious_blocks_taunt',teams=[[team(moves=['Taunt'],evs={'spe':252})],[team(ability='Oblivious')]]),
 dict(name='good_as_gold_blocks_taunt',teams=[[team(moves=['Taunt'],evs={'spe':252})],[team(ability='Good as Gold')]]),
 dict(name='magic_bounce_reflects_taunt',teams=[[team(moves=['Taunt'],evs={'spe':252})],[team(ability='Magic Bounce')]]),
 dict(name='bind_residual_expires_before_final_tick',initial=[{'volatiles':{'partiallytrapped':{'duration':3}}},{}],moves=[['Splash'],['Splash']],actions=[['move 1','move 1']]*3),
 dict(name='bind_ends_when_source_switches',teams=[[team()],[team(),team('Blissey')]],initial=[{'volatiles':{'partiallytrapped':{'duration':5}}},{}],actions=[['move 1','switch 2']]),
 dict(name='substitute_removes_bind',moves=[['Substitute'],['Splash']],initial=[{'volatiles':{'partiallytrapped':{'duration':5}}},{}]),
 dict(name='ghost_can_escape_bind',teams=[[team('Gengar'),team('Blissey')],[team()]],initial=[{'volatiles':{'partiallytrapped':{'duration':5}}},{}],actions=[['switch 2','move 1']]),
 dict(name='shed_shell_can_escape_bind',teams=[[team(item='Shed Shell'),team('Blissey')],[team()]],initial=[{'volatiles':{'partiallytrapped':{'duration':5}}},{}],actions=[['switch 2','move 1']]),
 dict(name='faint_replacement_does_not_give_free_attack',teams=[[team(moves=['Seismic Toss'],evs={'spe':252})],[team(),team('Blissey')]],initial=[{}, {'hp':100}],actions=[['move 1','move 1'],['','switch 2']]),
 dict(name='pivot_ko_chooses_before_foe_replacement',teams=[[team(moves=['U-turn'],evs={'spe':252}),team('Blissey')],[team(),team('Chansey')]],initial=[{}, {'hp':1}],actions=[['move 1','move 1'],['switch 2',''],['','switch 2']]),
 dict(name='pivot_resumes_queued_attack_on_incoming',teams=[[team('Mew',moves=['U-turn']),team('Blissey',item='Leftovers')],[team('Heatran',moves=['Seismic Toss'],ability='Shell Armor')]],initial=[{'stats':{'atk':1}},{}],actions=[['move 1','move 1'],['switch 2','']]),
 dict(name='slow_pivot_does_not_repeat_opponent_attack',teams=[[team('Mew',moves=['U-turn']),team('Blissey',item='Leftovers')],[team('Heatran',moves=['Seismic Toss'],ability='Shell Armor')]],initial=[{'stats':{'atk':1,'spe':1}},{}],actions=[['move 1','move 1'],['switch 2','']]),
 dict(name='replacement_hazard_ko_requires_another_choice',teams=[[team(moves=['Seismic Toss'],evs={'spe':252})],[team(),team('Shedinja',ability='Wonder Guard'),team('Blissey')]],initial=[{}, {'hp':100,'hazards':['stealthrock']}],actions=[['move 1','move 1'],['','switch 2'],['','switch 3']]),
 dict(name='rapid_spin_self_boost_and_cleanup_through_sub',moves=[['Rapid Spin'],['Splash']],initial=[{'hazards':['stealthrock']},{'volatiles':{'substitute':{'hp':1}}}]),
 dict(name='substitute_blocks_item_removal',teams=[[team(moves=['Knock Off'])],[team(item='Leftovers')]],initial=[{}, {'volatiles':{'substitute':{'hp':1}}}]),
 dict(name='substitute_blocks_secondary_nuzzle',moves=[['Nuzzle'],['Splash']],initial=[{}, {'volatiles':{'substitute':{'hp':1}}}]),
 dict(name='infiltrator_bypasses_substitute',teams=[[team(moves=['Seismic Toss'],ability='Infiltrator')],[team()]],initial=[{}, {'volatiles':{'substitute':{'hp':1}}}]),
 dict(name='sound_move_bypasses_substitute',teams=[[team('Mew',moves=['Hyper Voice'])],[team('Heatran',ability='Shell Armor')]],initial=[{'stats':{'spa':1}}, {'volatiles':{'substitute':{'hp':1}}}]),
 dict(name='drain_from_substitute_rounds_up',moves=[['Giga Drain'],['Splash']],initial=[{'hp':100},{'volatiles':{'substitute':{'hp':1}}}]),

 dict(name='mean_look_applies_source_bound_trap',moves=[['Mean Look'],['Splash']]),
 dict(name='mean_look_ghost_can_still_switch',teams=[[team(moves=['Mean Look'])],[team('Gengar'),team('Blissey')]],actions=[['move 1','move 1'],['move 1','switch 2']]),
 dict(name='confusion_expires_before_action',moves=[['Seismic Toss'],['Splash']],initial=[{'volatiles':{'confusion':{'time':1}}},{}]),
 dict(name='sleep_timer_runs_before_taunt',moves=[['Splash'],['Splash']],initial=[{'status':'slp','time':1,'volatiles':{'taunt':{'duration':2}}},{}]),
 dict(name='toxic_spikes_poison_entering_pokemon',teams=[[team(),team('Blissey')],[team()]],initial=[{'hazards':['toxicspikes']},{}],actions=[['switch 2','move 1']]),
 dict(name='poison_type_absorbs_toxic_spikes',teams=[[team(),team('Toxapex')],[team()]],initial=[{'hazards':['toxicspikes']},{}],actions=[['switch 2','move 1']]),
 dict(name='sticky_web_slows_replacement',teams=[[team(),team('Blissey')],[team()]],initial=[{'hazards':['stickyweb']},{}],actions=[['switch 2','move 1']]),
 dict(name='boots_block_web_and_poison',teams=[[team(),team('Blissey',item='Heavy-Duty Boots')],[team()]],initial=[{'hazards':['stickyweb','toxicspikes']},{}],actions=[['switch 2','move 1']]),
 dict(name='intimidate_on_entry',teams=[[team(),team('Gyarados',ability='Intimidate')],[team()]],actions=[['switch 2','move 1']]),
 dict(name='substitute_blocks_intimidate',teams=[[team(),team('Gyarados',ability='Intimidate')],[team()]],initial=[{}, {'volatiles':{'substitute':{'hp':115}}}],actions=[['switch 2','move 1']]),
 dict(name='clear_amulet_blocks_intimidate',teams=[[team(),team('Gyarados',ability='Intimidate')],[team(item='Clear Amulet')]],actions=[['switch 2','move 1']]),
 dict(name='defiant_reacts_to_intimidate',teams=[[team(),team('Gyarados',ability='Intimidate')],[team(ability='Defiant')]],actions=[['switch 2','move 1']]),
 dict(name='dauntless_shield_only_once_per_battle',teams=[[team(),team('Zamazenta',ability='Dauntless Shield')],[team()]],actions=[['switch 2','move 1'],['switch 2','move 1'],['switch 2','move 1']]),

 dict(name='two_pivots_keep_both_replacement_choices',teams=[[team('Mew',moves=['U-turn']),team('Ludicolo')],[team('Heatran',moves=['Flip Turn'],ability='Shell Armor'),team('Blissey')]],initial=[{'stats':{'atk':1}},{'stats':{'atk':1}}],actions=[['move 1','move 1'],['switch 2',''],['','switch 2']]),
 dict(name='simultaneous_faint_replacements_are_switch_only',teams=[[team(moves=['Tackle'],evs={'spe':252}),team('Blissey')],[team(),team('Chansey')]],initial=[{'hp':1,'pp':[0]},{'hp':1}],actions=[['move 1','move 1'],['switch 2','switch 2']]),

 dict(name='ghost_magic_bounce_reflects_mean_look',teams=[[team(moves=['Mean Look'])],[team('Gengar',ability='Magic Bounce')]]),
 dict(name='taunt_after_pivot_targets_a_fresh_entrant',teams=[[team('Mew',moves=['U-turn']),team('Blissey')],[team('Heatran',moves=['Taunt'],ability='Shell Armor')]],initial=[{'stats':{'atk':1}},{}],actions=[['move 1','move 1'],['switch 2','']]),

]

@pytest.fixture(scope='module')
def references():
 if not SHOWDOWN.exists(): pytest.skip('Install pinned official simulator')
 result=subprocess.run(['node',str(ROOT/'tests/fixtures/alignment_reference.cjs'),str(SHOWDOWN)],input=json.dumps(CASES),text=True,capture_output=True,check=True)
 return {row['name']:row for row in json.loads(result.stdout)}


def normalize_volatiles(mon,state):
 value=deepcopy(mon.volatiles)
 for key in ('partiallytrapped','trapped'):
  source=value.get(key,{}).get('source')
  if source:
   value[key]['source']=[source[0],(state.p1 if source[0]==1 else state.p2).pokemon[source[1]].species]
 return value

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_volatile_and_phase_trajectory(case,references):
 ref=references[case['name']];state=from_snapshot(ref['before'])
 for choices,expected in zip(case.get('actions',[['move 1','move 1']]),ref['results']):
  actions=[]
  for side_idx,(side,command) in enumerate(zip((state.p1,state.p2),choices)):
   if not command: actions.append(None);continue
   kind,slot=command.split();slot=int(slot)
   if kind=='switch':
    exp_side=expected['sides'][side_idx]
    species=exp_side['mons'][exp_side['active']]['species']
    index=next(i for i,p in enumerate(side.pokemon) if p.species==species)
    actions.append(SwitchAction(index+1,species))
   else: actions.append(MoveAction('struggle' if not any(m.pp for m in side.active_pokemon.moves) else side.active_pokemon.moves[slot-1].id,slot))
  state=simulate_turn_transition(state,*actions,sample_outcomes=True,rng=random.Random(17))
  for actual,exp in zip((state.p1,state.p2),expected['sides']):
   assert actual.active_pokemon.species==exp['mons'][exp['active']]['species']
   assert actual.hazards == from_snapshot(expected).__getattribute__('p1' if actual is state.p1 else 'p2').hazards
   for m in exp['mons']:
    ours=next(p for p in actual.pokemon if p.species==m['species'])
    assert ours.current_hp==m['hp'], (case['name'],choices,ours.species,ours.current_hp,m['hp'])
    if m['hp'] > 0: assert ours.status==STATUS[m['status']]
    assert ours.boosts==m['boosts']
    assert (ours.item or '')==m['item']
    assert [mv.pp for mv in ours.moves]==[mv['pp'] for mv in m['moves']]
    assert normalize_volatiles(ours,state)==m['volatiles']
  assert state.pending_switches==tuple(expected['pending'])
  assert state.turn==expected['turn']
  field=from_snapshot(expected)
  assert (state.weather,state.weather_turns,state.terrain,state.terrain_turns)==(field.weather,field.weather_turns,field.terrain,field.terrain_turns)


def test_taunt_and_trapping_filter_search_actions(references):
 state=from_snapshot(references['fast_taunt_cancels_pending_status_without_pp']['results'][0])
 assert all(a.action_type!=ActionType.MOVE or a.move_id=='seismictoss' for a in state.get_valid_actions(2))
 state=from_snapshot(references['bind_ends_when_source_switches']['before'])
 # Add a bench to the trapped side, verifying that the absence of switch actions
 # reflects the condition rather than an empty bench.
 state.p1.pokemon.append(state.p1.active_pokemon.clone())
 assert all(a.action_type!=ActionType.SWITCH for a in state.get_valid_actions(1))


@pytest.mark.parametrize('item',['','Choice Band'])
def test_confusion_frequency_and_self_damage_support(item):
 if not SHOWDOWN.exists(): pytest.skip('Install pinned official simulator')
 cases=[dict(name=str(i),teams=[[team(moves=['Seismic Toss'],item=item)],[team()]],
             initial=[{'volatiles':{'confusion':{'time':3}}},{}],seed=[i+1,7,8,9]) for i in range(512)]
 process=subprocess.run(['node',str(ROOT/'tests/fixtures/alignment_reference.cjs'),str(SHOWDOWN)],input=json.dumps(cases),text=True,capture_output=True,check=True)
 rows=json.loads(process.stdout);official=[];local=[]
 for i,row in enumerate(rows):
  before=from_snapshot(row['before'])
  after=simulate_turn_transition(before,MoveAction('seismictoss',1),MoveAction('splash',1),sample_outcomes=True,rng=random.Random(i))
  official.append(before.p1.active_pokemon.current_hp-row['results'][0]['sides'][0]['mons'][0]['hp'])
  local.append(before.p1.active_pokemon.current_hp-after.p1.active_pokemon.current_hp)
  if local[-1]:assert after.p1.active_pokemon.moves[0].pp==before.p1.active_pokemon.moves[0].pp
 assert set(official)==set(local)
 assert .25<sum(x>0 for x in official)/512<.42
 assert .25<sum(x>0 for x in local)/512<.42
 assert abs(sum(x>0 for x in official)-sum(x>0 for x in local))/512<.08


@pytest.mark.parametrize('item',['Grip Claw','Binding Band'])
def test_magma_storm_damage_duration_and_accuracy_against_official(item):
 if not SHOWDOWN.exists(): pytest.skip('Install pinned official simulator')
 cases=[dict(name=str(i),teams=[[team(moves=['Magma Storm'],item=item)],[team(ability='Shell Armor')]],seed=[i+1,7,8,9]) for i in range(256)]
 process=subprocess.run(['node',str(ROOT/'tests/fixtures/alignment_reference.cjs'),str(SHOWDOWN)],input=json.dumps(cases),text=True,capture_output=True,check=True)
 rows=json.loads(process.stdout);damages=[[],[]];durations=[[],[]]
 for i,row in enumerate(rows):
  before=from_snapshot(row['before'])
  after=simulate_turn_transition(before,MoveAction('magmastorm',1),MoveAction('splash',1),sample_outcomes=True,rng=random.Random(i))
  expected=row['results'][0]['sides'][1]['mons'][0]
  damages[0].append(before.p2.active_pokemon.current_hp-expected['hp'])
  damages[1].append(before.p2.active_pokemon.current_hp-after.p2.active_pokemon.current_hp)
  durations[0].append(expected['volatiles'].get('partiallytrapped',{}).get('duration',0))
  durations[1].append(after.p2.active_pokemon.volatiles.get('partiallytrapped',{}).get('duration',0))
 assert set(damages[0])==set(damages[1])
 assert set(durations[0])==set(durations[1])==({0,7} if item=='Grip Claw' else {0,4,5})
 assert abs(sum(x>0 for x in damages[0])-sum(x>0 for x in damages[1]))/256<.10


def test_pending_branches_preserve_queue_and_waiting_player(references):
 from glaubermon.search.subgame_resolver import SubgameResolver
 state=from_snapshot(references['pivot_resumes_queued_attack_on_incoming']['before'])
 phase=simulate_turn_transition(state,MoveAction('uturn',1),MoveAction('seismictoss',1),sample_outcomes=True,rng=random.Random(5))
 assert phase.pending_switches==(1,) and phase.continuation
 assert phase.get_valid_actions(2)==[]
 original_pp=phase.p2.active_pokemon.moves[0].pp
 a=SwitchAction(2,'Blissey')
 one=simulate_turn_transition(phase,a,None)
 two=simulate_turn_transition(phase,a,None)
 assert one==two and not one.pending_switches
 assert one.p2.active_pokemon.moves[0].pp==original_pp-1
 assert phase.p2.active_pokemon.moves[0].pp==original_pp
 assert phase.p1.active_index==0
 with pytest.raises(ValueError,match='must wait'):
  simulate_turn_transition(phase,a,MoveAction('seismictoss',1))
 resolver=SubgameResolver()
 action,prob,actions,value=resolver._resolve_replacements(phase,0,False)
 assert action==a and len(actions)==len(prob)==1
 assert phase.continuation and phase.p1.active_index==0


def test_perspective_flip_preserves_trap_sources_and_pending_queue(references):
 state=from_snapshot(references['pivot_resumes_queued_attack_on_incoming']['before'])
 state.p1.active_pokemon.volatiles['partiallytrapped']={'duration':5,'source':(2,0),'divisor':8}
 phase=simulate_turn_transition(state,MoveAction('uturn',1),MoveAction('seismictoss',1))
 flipped=phase.flipped()
 assert flipped.pending_switches==(2,)
 assert flipped.p2.active_pokemon.volatiles['partiallytrapped']['source']==(1,0)
 assert flipped.flipped()==phase
 from_first=simulate_turn_transition(phase,SwitchAction(2,'Blissey'),None)
 from_second=simulate_turn_transition(flipped,None,SwitchAction(2,'Blissey')).flipped()
 assert from_first==from_second
 assert phase.p1.active_index==0


def test_all_replacement_options_are_evaluated_in_one_batch(references):
 import numpy as np
 from glaubermon.search.subgame_resolver import SubgameResolver
 state=from_snapshot(references['faint_replacement_does_not_give_free_attack']['before'])
 state.p2.pokemon.append(state.p2.pokemon[1].clone())
 state.p2.pokemon[-1].species='Chansey'
 state.p2.pokemon[-1].current_hp=50
 phase=simulate_turn_transition(state,MoveAction('seismictoss',1),MoveAction('splash',1))
 class BatchOnly:
  def __init__(self):self.sizes=[]
  def evaluate_batch(self,states):
   self.sizes.append(len(states))
   return np.array([-child.p2.active_pokemon.hp_percent for child in states])
  def evaluate(self,state):raise AssertionError('Expected one batched evaluation')
 evaluator=BatchOnly();resolver=SubgameResolver(evaluator)
 result=resolver._resolve_replacements(phase,0,False,True)
 assert evaluator.sizes==[2]
 assert len(result[6])==2 and result[4].target_slot==2
 assert result[3]==-1.0
