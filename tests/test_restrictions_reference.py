"""Differential action restrictions and slot-bound residuals, Gen 9."""
import json,subprocess,random
import pytest
from test_alignment_reference import ROOT,SHOWDOWN,from_snapshot
from test_volatile_phases_reference import test_volatile_and_phase_trajectory as _compare
from test_field_multihit_reference import team
from glaubermon.core.actions import MoveAction
from glaubermon.search.subgame_resolver import simulate_turn_transition

CASES=[]
for move in ('Encore','Disable'):
 for fast in (True,False):
  CASES.append(dict(name=move+str(fast),teams=[[team(moves=['Splash',move],evs={'spe':252} if fast else {})],[team(moves=['Splash','Seismic Toss'],evs={} if fast else {'spe':252})]],actions=[['move 1','move 1'],['move 2','move 2']]+[['move 1','move 2' if ((move=='Disable')==fast) else 'move 1']]*4))
 CASES.extend([
  dict(name=move+'_no_previous_move',moves=[[move],['Splash']],initial=[{'stats':{'spe':400}},{}]),
  dict(name=move+'_no_pp',moves=[[move],['Splash']],initial=[{}, {'last_move':'Splash','pp':[0]}]),
  dict(name=move+'_mental_herb',teams=[[team(moves=[move])],[team(item='Mental Herb')]],initial=[{}, {'last_move':'Splash'}]),
  dict(name=move+'_aroma_veil',teams=[[team(moves=[move])],[team(ability='Aroma Veil')]],initial=[{}, {'last_move':'Splash'}]),
  dict(name=move+'_bypasses_substitute',moves=[[move],['Splash']],initial=[{}, {'last_move':'Splash','volatiles':{'substitute':{'hp':50}}}]),
  dict(name=move+'_switch_clears',teams=[[team(moves=[move,'Splash'])],[team(),team('Blissey')]],initial=[{}, {'last_move':'Splash'}],actions=[['move 1','move 1'],['move 2','switch 2']]),
  dict(name=move+'_magic_bounce',teams=[[team(moves=[move,'Splash'])],[team(ability='Magic Bounce')]],initial=[{'last_move':'Splash'},{}]),
 ])
CASES += [
 dict(name='disable_cancels_pending_without_pp',teams=[[team('Mew',moves=['Disable'])],[team(moves=['Seismic Toss','Splash'])]],initial=[{}, {'last_move':'Seismic Toss'}]),
 dict(name='encore_expires_on_last_pp',teams=[[team('Mew',moves=['Encore'])],[team(moves=['Seismic Toss','Splash'])]],initial=[{}, {'last_move':'Seismic Toss','pp':[1,64]}]),
 dict(name='encore_cannot_lock_encore',moves=[['Encore'],['Splash']],initial=[{}, {'last_move':'Encore'}]),
 dict(name='leechseed_normal',moves=[['Leech Seed','Splash'],['Splash']],initial=[{'hp':200},{}],actions=[['move 1','move 1']]+[['move 2','move 1']]*2),
 dict(name='leechseed_grass_immunity',teams=[[team(moves=['Leech Seed'])],[team('Venusaur')]]),
 dict(name='leechseed_substitute_blocks',moves=[['Leech Seed'],['Splash']],initial=[{}, {'volatiles':{'substitute':{'hp':50}}}]),
 dict(name='leechseed_magic_guard',teams=[[team(moves=['Leech Seed'])],[team(ability='Magic Guard')]],initial=[{'hp':200},{}]),
 dict(name='leechseed_liquid_ooze',teams=[[team(moves=['Leech Seed'])],[team(ability='Liquid Ooze')]],initial=[{'hp':200},{}]),
 dict(name='leechseed_big_root',teams=[[team(moves=['Leech Seed'],item='Big Root')],[team()]],initial=[{'hp':200},{}]),
 dict(name='leechseed_small_remaining_hp',moves=[['Leech Seed'],['Splash']],initial=[{'hp':200},{'hp':10}]),
 dict(name='leechseed_magic_bounce',teams=[[team(moves=['Leech Seed'])],[team(ability='Magic Bounce')]],initial=[{}, {'hp':200}]),
 dict(name='leechseed_follows_source_slot',teams=[[team(moves=['Leech Seed','Splash']),team('Blissey')],[team(moves=['Splash','Seismic Toss'])]],actions=[['move 1','move 1'],['switch 2','move 2']]),
 dict(name='leechseed_switch_target_clears',teams=[[team(moves=['Leech Seed','Splash'])],[team(),team('Blissey')]],actions=[['move 1','move 1'],['move 2','switch 2']]),
 dict(name='leechseed_rapid_spin_clears',teams=[[team('Mew',moves=['Leech Seed'])],[team(moves=['Rapid Spin'])]],initial=[{}, {'stats':{'atk':1}}]),
 dict(name='leechseed_survives_new_substitute',teams=[[team('Mew',moves=['Leech Seed'])],[team(moves=['Substitute'])]]),
 dict(name='leechseed_before_poison',teams=[[team(moves=['Splash'],item='Leftovers')],[team()]],initial=[{'hp':40,'status':'psn','volatiles':{'leechseed':{}}},{'hp':100}]),
]

CASES += [
 dict(name='encore_and_disable_force_struggle',refresh_disabled=True,teams=[[team()],[team(moves=['Seismic Toss','Splash'])]],initial=[{}, {'last_move':'Seismic Toss','stats':{'atk':1,'spe':400},'pp':[32,64],'volatiles':{'encore':{'duration':2},'disable':{'duration':2}}}]),
 dict(name='encore_keeps_selected_priority',terrain='psychicterrain',teams=[[team('Mew',moves=['Encore'],ability='Prankster')],[team(moves=['Splash','Quick Attack'])]],initial=[{}, {'last_move':'Splash'}],actions=[['move 1','move 2']]),
 dict(name='leechseed_does_not_heal_fainted_slot',teams=[[team(moves=['Leech Seed']),team('Blissey')],[team('Mew',moves=['Seismic Toss'])]],initial=[{'hp':100},{}]),
 dict(name='leechseed_grassy_and_leftovers_order',terrain='grassyterrain',teams=[[team(item='Leftovers')],[team()]],initial=[{'hp':40,'status':'psn','volatiles':{'leechseed':{}}},{'hp':100}]),
]

for reverse in (False,True):
 CASES.append(dict(name='double_seed_speed_order'+str(reverse),teams=[[team()],[team('Mew')]],trick_room=reverse,initial=[{'hp':40,'volatiles':{'leechseed':{}}},{'hp':40,'volatiles':{'leechseed':{}}}]))

# Avoid RNG-dependent speed ties in deterministic before/after assertions.
for case in CASES:
 case['compare_last_move']=True
 if case['name'].startswith(('Encore_','Disable_','encore_cannot')):
  case.setdefault('initial',[{},{}])[0].setdefault('stats',{})['spe']=400

@pytest.fixture(scope='module')
def references():
 if not SHOWDOWN.exists():pytest.skip('Install pinned official simulator')
 p=subprocess.run(['node',str(ROOT/'tests/fixtures/alignment_reference.cjs'),str(SHOWDOWN)],input=json.dumps(CASES),capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 return {r['name']:r for r in json.loads(p.stdout)}

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_restriction_trajectory(case,references):
 _compare(case,references)


def test_leech_seed_hit_frequency():
 cases=[dict(name=str(i),moves=[['Leech Seed'],['Splash']],seed=[i+1,2,3,4]) for i in range(512)]
 if not SHOWDOWN.exists():pytest.skip('Install pinned official simulator')
 rows=json.loads(subprocess.run(['node',str(ROOT/'tests/fixtures/alignment_reference.cjs'),str(SHOWDOWN)],input=json.dumps(cases),capture_output=True,text=True,check=True).stdout)
 official=local=0
 for i,row in enumerate(rows):
  state=from_snapshot(row['before'])
  after=simulate_turn_transition(state,MoveAction('leechseed',1),MoveAction('splash',1),sample_outcomes=True,rng=random.Random(i))
  official+='leechseed' in row['results'][0]['sides'][1]['mons'][0]['volatiles']
  local+='leechseed' in after.p2.active_pokemon.volatiles
 assert .84<official/512<.96 and .84<local/512<.96
 assert abs(official-local)/512<.08
