"""Differential tests: actual official transitions, not hand-written desired outcomes."""
import json, random, subprocess
from pathlib import Path
from copy import deepcopy
import pytest
from glaubermon.core.battle_state import BattleState,BattleSide
from glaubermon.core.pokemon import Pokemon,Move
from glaubermon.core.actions import MoveAction,SwitchAction
from glaubermon.core.types import PokemonType,StatusCondition,Hazard,Weather,Terrain
from glaubermon.search.subgame_resolver import simulate_turn_transition

ROOT=Path(__file__).resolve().parents[1]
SHOWDOWN=ROOT.parent/'showdown-parity/node_modules/pokemon-showdown'
if not SHOWDOWN.exists(): SHOWDOWN=ROOT/'tools/showdown/node_modules/pokemon-showdown'
CASES=[
 {'name':'wake_and_act','initial':[{'status':'slp','time':1},{}]},
 {'name':'still_asleep','initial':[{'status':'slp','time':2},{}]},
 {'name':'sleep_talk','moves':[['Sleep Talk','Seismic Toss'],['Splash']], 'initial':[{'status':'slp','time':2},{}]},
 {'name':'awake_sleep_talk_fails','moves':[['Sleep Talk','Seismic Toss'],['Splash']]},
 {'name':'rest_two_missed_turns','moves':[['Rest','Seismic Toss'],['Splash']], 'initial':[{'hp':200},{}],
  'actions':[['move 1','move 1'],['move 2','move 1'],['move 2','move 1'],['move 2','move 1']]},
 {'name':'reflect_cast','moves':[['Reflect'],['Seismic Toss']]},
 {'name':'light_screen_cast','moves':[['Light Screen'],['Seismic Toss']]},
 {'name':'tailwind_cast','moves':[['Tailwind'],['Seismic Toss']]},
 {'name':'trick_room_cast','moves':[['Trick Room'],['Seismic Toss']]},
 {'name':'trick_room_toggle','moves':[['Trick Room'],['Splash']],'trick_room':True},
 {'name':'seismic_toss_ghost','teams':[[{'species':'Snorlax','moves':['Seismic Toss']}],[{'species':'Gengar','moves':['Splash']}]]},
 {'name':'night_shade_normal','moves':[['Night Shade'],['Splash']]},
 {'name':'immune_volt_switch','teams':[[{'species':'Snorlax','moves':['Volt Switch']},{'species':'Blissey','moves':['Splash']}],[{'species':'Garchomp','moves':['Splash']}]]},
 {'name':'switch_resets_boosts','teams':[[{'species':'Snorlax','moves':['Splash']},{'species':'Blissey','moves':['Splash']}],[{'species':'Snorlax','moves':['Splash']}]],
  'initial':[{'boosts':{'atk':4,'def':2}},{}],'actions':[['switch 2','move 1'],['switch 2','move 1']]},
 {'name':'toxic_progression','moves':[['Splash'],['Splash']],'initial':[{'status':'tox'},{}],
  'actions':[['move 1','move 1']]*3},
 {'name':'toxic_orb_delay','teams':[[{'species':'Snorlax','item':'Toxic Orb','moves':['Splash']}],[{'species':'Snorlax','moves':['Splash']}]],'actions':[['move 1','move 1']]*2},
]
for key in ('trick_room','tailwind'):
 CASES.append({'name':key+'_order','teams':[[{'species':'Snorlax','moves':['Seismic Toss']}],[{'species':'Snorlax','moves':['Seismic Toss'],'evs':{'spe':252}}]],
 'initial':[{'hp':100,**({'tailwind':True} if key=='tailwind' else {})},{'hp':100}], 'trick_room':key=='trick_room'})

STATUS={'fnt':StatusCondition.NONE,'':StatusCondition.NONE,'slp':StatusCondition.SLEEP,'tox':StatusCondition.TOXIC,'brn':StatusCondition.BURN,'par':StatusCondition.PARALYSIS,'psn':StatusCondition.POISON,'frz':StatusCondition.FREEZE}

def from_snapshot(data):
 sides=[]
 hazard_keys={"stealthrock":Hazard.STEALTH_ROCK,"spikes":Hazard.SPIKES_1,"toxicspikes":Hazard.TOXIC_SPIKES_1,"stickyweb":Hazard.STICKY_WEB}
 for side in data['sides']:
  mons=[]
  for p in side['mons']:
   moves=[]
   for m in p['moves']:
    move=Move.from_dex(m['id']);move.pp=m['pp'];move.max_pp=m['maxpp'];moves.append(move)
   types=[PokemonType(t) for t in p['types']]
   mons.append(Pokemon(species=p['species'],types=(types[0],types[1] if len(types)>1 else None),current_hp=p['hp'],max_hp=p['maxhp'],
      level=p.get('level',100),volatiles=deepcopy(p.get('volatiles',{})),raw_stats={'hp':p['maxhp'],**p['stats']},ability=p['ability'],item=p['item'],status=STATUS[p['status']],status_turns=p['time'],boosts=p['boosts'].copy(),moves=moves))
   mons[-1].shield_boosted=p.get('shieldBoost',False)
   mons[-1].sword_boosted=p.get('swordBoost',False)
  sides.append(BattleSide(mons,active_index=side['active'],hazards={hazard_keys[k]:v for k,v in side.get('hazards',{}).items()},screens=side['screens'].copy(),tailwind=side['tailwind']))
 for side in sides:
  for mon in side.pokemon:
   for key in ('trapped','partiallytrapped'):
    source=mon.volatiles.get(key,{}).get('source')
    if source:
     mon.volatiles[key]['source']=(source[0],next(i for i,p in enumerate(sides[source[0]-1].pokemon) if p.species==source[1]))
 return BattleState(*sides,weather={'sunnyday':Weather.SUN,'raindance':Weather.RAIN,'sandstorm':Weather.SANDSTORM,'snowscape':Weather.SNOW}.get(data.get('weather'),Weather.NONE),weather_turns=data.get('weather_turns',0),terrain={t.value+'terrain':t for t in Terrain}.get(data.get('terrain'),Terrain.NONE),terrain_turns=data.get('terrain_turns',0),trick_room=data['trick_room'],turn=data.get('turn',1),pending_switches=tuple(data.get('pending',[])))

@pytest.fixture(scope='module')
def reference():
 if not SHOWDOWN.exists():pytest.skip('Install pinned official simulator')
 p=subprocess.run(['node',str(ROOT/'tests/fixtures/alignment_reference.cjs'),str(SHOWDOWN)],input=json.dumps(CASES),text=True,capture_output=True,check=True)
 return {r['name']:r for r in json.loads(p.stdout)}

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_official_transition_parity(case,reference):
 ref=reference[case['name']]; state=from_snapshot(ref['before'])
 for choices,expected in zip(case.get('actions',[['move 1','move 1']]),ref['results']):
  actions=[]
  for side,command in zip((state.p1,state.p2),choices):
   if not command:
    actions.append(None);continue
   kind,slot=command.split();slot=int(slot)
   # Official side order swaps the active to slot 1; our arrays preserve slots.
   if kind=='switch':
    species=expected['sides'][len(actions)]['mons'][expected['sides'][len(actions)]['active']]['species']
    index=next(i for i,p in enumerate(side.pokemon) if p.species==species)
    actions.append(SwitchAction(index+1,species))
   else:actions.append(MoveAction(side.active_pokemon.moves[slot-1].id,slot))
  state=simulate_turn_transition(state,*actions,sample_outcomes=True,rng=random.Random(17))
  for actual,exp in zip((state.p1,state.p2),expected['sides']):
   assert actual.active_pokemon.species==exp['mons'][exp['active']]['species']
   for m in exp['mons']:
    ours=next(p for p in actual.pokemon if p.species==m['species'])
    assert ours.current_hp==m['hp']
    assert ours.status==STATUS[m['status']]
    assert ours.boosts==m['boosts']
    assert [mv.pp for mv in ours.moves]==[mv['pp'] for mv in m['moves']]
   assert actual.screens==exp['screens']
   assert actual.tailwind==exp['tailwind']
  assert state.pending_switches==tuple(expected.get('pending',[]))
  assert state.turn==expected.get('turn',state.turn)
  assert state.trick_room==expected['trick_room']

@pytest.mark.parametrize('status,low,high',[('par',0.17,0.33),('frz',0.70,0.89)])
def test_status_failure_frequencies_against_official(status,low,high):
 if not SHOWDOWN.exists():pytest.skip('Install pinned official simulator')
 cases=[dict(name=str(i),initial=[{'status':status},{}],seed=[i+1,2,3,4]) for i in range(512)]
 process=subprocess.run(['node',str(ROOT/'tests/fixtures/alignment_reference.cjs'),str(SHOWDOWN)],input=json.dumps(cases),text=True,capture_output=True,check=True)
 official=json.loads(process.stdout); failures=[]; local=[]
 for i,row in enumerate(official):
  failures.append(row['results'][0]['sides'][1]['mons'][0]['hp']==row['before']['sides'][1]['mons'][0]['hp'])
  state=from_snapshot(row['before'])
  after=simulate_turn_transition(state,MoveAction('seismictoss',1),MoveAction('splash',1),sample_outcomes=True,rng=random.Random(i))
  local.append(after.p2.active_pokemon.current_hp==state.p2.active_pokemon.current_hp)
 assert low<sum(failures)/512<high
 assert low<sum(local)/512<high
 assert abs(sum(failures)-sum(local))/512<0.10

@pytest.mark.parametrize('move,screens,ability,critical',[
 ('Tackle',['reflect'],'Thick Fat',False),('Swift',['lightscreen'],'Thick Fat',False),
 ('Tackle',['reflect','auroraveil'],'Thick Fat',False),
 ('Tackle',['reflect'],'Infiltrator',False),('Wicked Blow',['reflect'],'Thick Fat',True)])
def test_screen_damage_support_against_official(move,screens,ability,critical):
 from glaubermon.inference.damage_calc import calculate_damage_rolls
 if not SHOWDOWN.exists():pytest.skip('Install pinned official simulator')
 cases=[dict(name=str(i),teams=[[{'species':'Snorlax','ability':ability,'moves':[move]}],
 [{'species':'Snorlax','ability':'Thick Fat' if critical else 'Shell Armor','moves':['Splash']}]],
 initial=[{}, {'screens':screens}],seed=[i+1,2,3,4]) for i in range(256)]
 process=subprocess.run(['node',str(ROOT/'tests/fixtures/alignment_reference.cjs'),str(SHOWDOWN)],input=json.dumps(cases),text=True,capture_output=True,check=True)
 official=json.loads(process.stdout)
 state=from_snapshot(official[0]['before'])
 actual=calculate_damage_rolls(state.p1.active_pokemon,state.p2.active_pokemon,state.p1.active_pokemon.moves[0],
                             defender_side=state.p2,is_critical=critical)
 damages={r['before']['sides'][1]['mons'][0]['hp']-r['results'][0]['sides'][1]['mons'][0]['hp'] for r in official}
 assert damages==set(actual)

@pytest.mark.parametrize('move,ability,item,status,low,high',[
 ('Nuzzle','Thick Fat','','par',1.,1.),('Nuzzle','Shield Dust','','par',0.,0.),
 ('Nuzzle','Thick Fat','Covert Cloak','par',0.,0.),('Thunderbolt','Thick Fat','','par',.03,.18),
 ('Mortal Spin','Thick Fat','','psn',1.,1.),('Mortal Spin','Shield Dust','','psn',0.,0.),
 ('Mortal Spin','Thick Fat','Covert Cloak','psn',0.,0.)])
def test_secondary_status_against_official(move,ability,item,status,low,high):
 if not SHOWDOWN.exists():pytest.skip('Install pinned official simulator')
 cases=[dict(name=str(i),teams=[[{'species':'Snorlax','moves':[move]}],
 [{'species':'Snorlax','ability':ability,'item':item,'moves':['Splash']}]],seed=[i+1,3,4,5]) for i in range(256)]
 process=subprocess.run(['node',str(ROOT/'tests/fixtures/alignment_reference.cjs'),str(SHOWDOWN)],input=json.dumps(cases),text=True,capture_output=True,check=True)
 official=json.loads(process.stdout);counts=[0,0]
 for i,row in enumerate(official):
  state=from_snapshot(row['before'])
  after=simulate_turn_transition(state,MoveAction(state.p1.active_pokemon.moves[0].id,1),MoveAction('splash',1),sample_outcomes=True,rng=random.Random(i))
  counts[0]+=row['results'][0]['sides'][1]['mons'][0]['status']==status
  counts[1]+=after.p2.active_pokemon.status==STATUS[status]
 assert all(low<=n/256<=high for n in counts)
 assert abs(counts[0]-counts[1])/256<.10
