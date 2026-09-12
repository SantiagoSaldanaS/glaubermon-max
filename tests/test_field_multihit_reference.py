"""Weather, terrain, speed and per-strike rules against pinned Gen 9 Showdown."""
import json,random,subprocess
import pytest
from test_alignment_reference import ROOT,SHOWDOWN,from_snapshot
from test_volatile_phases_reference import test_volatile_and_phase_trajectory as _compare
from glaubermon.core.actions import MoveAction
from glaubermon.search.subgame_resolver import simulate_turn_transition
from glaubermon.inference.damage_calc import calculate_damage_rolls


def team(species='Snorlax',moves=None,**kwargs):
 return dict(species=species,moves=moves or ['Splash'],ability='Shell Armor',**kwargs) if 'ability' not in kwargs else dict(species=species,moves=moves or ['Splash'],**kwargs)

CASES=[]
for weather,move,ability,item in [('raindance','Rain Dance','Drizzle','Damp Rock'),('sunnyday','Sunny Day','Drought','Heat Rock'),('sandstorm','Sandstorm','Sand Stream','Smooth Rock'),('snowscape','Snowscape','Snow Warning','Icy Rock')]:
 for held in ('',item):
  CASES.append(dict(name=move+held,teams=[[team(moves=[move,'Splash'],item=held)],[team()]],actions=[['move 1','move 1']]+[['move 2','move 1']]*8))
 CASES.append(dict(name=ability,teams=[[team(),team('Blissey',ability=ability,item=item)],[team()]],actions=[['switch 2','move 1']]))
for terrain,move,ability in [('electricterrain','Electric Terrain','Electric Surge'),('grassyterrain','Grassy Terrain','Grassy Surge'),('psychicterrain','Psychic Terrain','Psychic Surge'),('mistyterrain','Misty Terrain','Misty Surge')]:
 for held in ('','Terrain Extender'):
  CASES.append(dict(name=move+held,teams=[[team(moves=[move,'Splash'],item=held)],[team()]],initial=[{'hp':100},{'hp':200}],actions=[['move 1','move 1']]+[['move 2','move 1']]*8))
 CASES.append(dict(name=ability,teams=[[team(),team('Blissey',ability=ability)],[team()]],actions=[['switch 2','move 1']]))
for ability,weather in [('Rain Dish','raindance'),('Dry Skin','raindance'),('Dry Skin','sunnyday'),('Solar Power','sunnyday'),('Ice Body','snowscape'),('Overcoat','sandstorm'),('Sand Rush','sandstorm'),('Magic Guard','sandstorm')]:
 CASES.append(dict(name=ability+weather,weather=weather,teams=[[team(ability=ability,item='Leftovers')],[team()]],initial=[{'hp':200},{}]))
CASES += [
 dict(name='cloudnine_suppresses_sand',weather='sandstorm',teams=[[team(ability='Cloud Nine')],[team()]]),
 dict(name='sand_ends_before_chip',weather='sandstorm',weather_turns=1),
 dict(name='grassy_heals_before_expiry',terrain='grassyterrain',terrain_turns=1,initial=[{'hp':200},{'hp':200}]),
 dict(name='flying_ignores_grassy',terrain='grassyterrain',teams=[[team('Pelipper')],[team()]],initial=[{'hp':100},{'hp':200}]),
 dict(name='electric_blocks_rest',terrain='electricterrain',moves=[['Rest'],['Splash']],initial=[{'hp':200},{}]),
 dict(name='misty_blocks_rest',terrain='mistyterrain',moves=[['Rest'],['Splash']],initial=[{'hp':200},{}]),
 dict(name='psychic_blocks_priority',terrain='psychicterrain',moves=[['Quick Attack'],['Splash']]),
 dict(name='prankster_psychic_block',terrain='psychicterrain',teams=[[team(moves=['Taunt'],ability='Prankster')],[team()]]),
]
for ability,field in [('Swift Swim',{'weather':'raindance'}),('Chlorophyll',{'weather':'sunnyday'}),('Sand Rush',{'weather':'sandstorm'}),('Slush Rush',{'weather':'snowscape'}),('Surge Surfer',{'terrain':'electricterrain'}),('Quick Feet',{})]:
 for reverse in (False,True):
  CASES.append(dict(name=ability+str(reverse),**field,trick_room=reverse,teams=[[team(moves=['Seismic Toss'],ability=ability)],[team(moves=['Seismic Toss'])]],initial=[{'hp':100,'stats':{'spe':101},**({'status':'par'} if ability=='Quick Feet' else {})},{'hp':100,'stats':{'spe':140}}]))
for move in ('Synthesis','Moonlight','Morning Sun'):
 for weather in ('','sunnyday','raindance','sandstorm'):
  CASES.append(dict(name=move+weather,weather=weather,teams=[[team(moves=[move])],[team()]],initial=[{'hp':100},{}]))
for move in ('Will-O-Wisp','Toxic','Thunder Wave'):
 CASES.append(dict(name=move+'_misty',terrain='mistyterrain',moves=[[move],['Splash']]))
CASES.append(dict(name='ice_spinner_removes_terrain_timer',terrain='grassyterrain',teams=[[team('Mew',moves=['Ice Spinner'])],[team()]],initial=[{'stats':{'atk':1}},{}]))
CASES.append(dict(name='prankster_dark_immunity',teams=[[team(moves=['Taunt'],ability='Prankster')],[team('Umbreon')]]))
for move in ('Double Hit','Triple Axel','Triple Kick','Triple Dive','Bullet Seed','Population Bomb'):
 CASES.append(dict(name=move+'_helmet_stops_hits',teams=[[team('Mew',moves=[move],ability='Skill Link')],[team(item='Rocky Helmet')]],initial=[{'hp':100,'stats':{'atk':1}},{}]))
CASES += [dict(name='multi_breaks_sub_then_hits_target',teams=[[team('Mew',moves=['Triple Axel'],item='Loaded Dice')],[team()]],initial=[{'stats':{'atk':1}},{'volatiles':{'substitute':{'hp':1}}}]),
 dict(name='multi_lifeorb_once',teams=[[team('Mew',moves=['Bullet Seed'],ability='Skill Link',item='Life Orb')],[team()]],initial=[{'stats':{'atk':1}},{}])]


def run(cases):
 if not SHOWDOWN.exists():pytest.skip('Install pinned official simulator')
 p=subprocess.run(['node',str(ROOT/'tests/fixtures/alignment_reference.cjs'),str(SHOWDOWN)],input=json.dumps(cases),text=True,capture_output=True,check=True)
 return json.loads(p.stdout)

@pytest.fixture(scope='module')
def references():return {r['name']:r for r in run(CASES)}

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_field_and_multihit_trajectory(case,references):
 _compare(case,references)
 # Reuse the same trajectory runner with field assertions enabled below.

@pytest.mark.parametrize('move,weather,terrain,species,ability',[
 ('Surf','raindance','','Snorlax','Shell Armor'),('Flamethrower','raindance','','Snorlax','Shell Armor'),
 ('Thunderbolt','','electricterrain','Snorlax','Shell Armor'),('Energy Ball','','grassyterrain','Snorlax','Shell Armor'),
 ('Psychic','','psychicterrain','Snorlax','Shell Armor'),('Dragon Pulse','','mistyterrain','Snorlax','Shell Armor'),
 ('Earthquake','','grassyterrain','Snorlax','Shell Armor'),('Swift','sandstorm','','Tyranitar','Shell Armor'),
 ('Tackle','snowscape','','Glaceon','Shell Armor'),('Weather Ball','raindance','','Snorlax','Shell Armor'),
 ('Weather Ball','sunnyday','','Snorlax','Cloud Nine'),('Terrain Pulse','','electricterrain','Snorlax','Shell Armor')])
def test_damage_support(move,weather,terrain,species,ability):
 rows=run([dict(name=str(i),weather=weather,terrain=terrain,teams=[[team('Mew',moves=[move])],[team(species,ability=ability,item='Safety Goggles')]],seed=[i+1,2,3,4]) for i in range(256)])
 state=from_snapshot(rows[0]['before']);a,d=state.p1.active_pokemon,state.p2.active_pokemon
 rolls=calculate_damage_rolls(a,d,a.moves[0],state.weather,state.terrain)
 observed={d['damage'] for r in rows for d in r['damageTrace'] if d['side']==2 and d['effect']==a.moves[0].id and not d['critical']}
 assert observed==set(rolls)

@pytest.mark.parametrize('move,item,ability',[('Bullet Seed','','Shell Armor'),('Bullet Seed','Loaded Dice','Shell Armor'),('Bullet Seed','','Skill Link'),('Triple Axel','','Shell Armor'),('Triple Axel','Loaded Dice','Shell Armor'),('Population Bomb','Loaded Dice','Shell Armor')])
def test_hit_count_distribution(move,item,ability,monkeypatch):
 from glaubermon.core.pokemon import Pokemon
 damage=Pokemon.take_damage
 strikes=[]
 def trace(mon,amount):
  if mon.species=="Snorlax" and amount>0:strikes.append(amount)
  return damage(mon,amount)
 monkeypatch.setattr(Pokemon,"take_damage",trace)
 rows=run([dict(name=str(i),teams=[[team('Mew',moves=[move],item=item,ability=ability)],[team()]],initial=[{'stats':{'atk':1}},{}],seed=[i+1,2,3,4]) for i in range(512)])
 official=[];local=[]
 for i,row in enumerate(rows):
  state=from_snapshot(row['before'])
  strikes.clear()
  after=simulate_turn_transition(state,MoveAction(state.p1.active_pokemon.moves[0].id,1),MoveAction('splash',1),sample_outcomes=True,rng=random.Random(i))
  official.append(len([d for d in row['damageTrace'] if d['side']==2 and d['effect']==state.p1.active_pokemon.moves[0].id]))
  local.append(len(strikes))
 assert set(local)==set(official)
 for damage in set(official):assert abs(official.count(damage)-local.count(damage))/len(rows)<.10


def test_triple_axel_per_strike_damage(monkeypatch):
 from glaubermon.core.pokemon import Pokemon
 original=Pokemon.take_damage
 strikes=[]
 def trace(mon,amount):
  if mon.species=="Snorlax" and amount>0:strikes.append(amount)
  return original(mon,amount)
 monkeypatch.setattr(Pokemon,"take_damage",trace)
 rows=run([dict(name=str(i),teams=[[team('Mew',moves=['Triple Axel'],item='Loaded Dice')],[team()]],seed=[i+1,2,3,4]) for i in range(256)])
 state=from_snapshot(rows[0]['before'])
 local=[set(),set(),set()]
 for i in range(256):
  strikes.clear()
  simulate_turn_transition(state,MoveAction('tripleaxel',1),MoveAction('splash',1),sample_outcomes=True,rng=random.Random(i))
  for hit,damage in enumerate(strikes):local[hit].add(damage)
 for hit,power in enumerate((20,40,60)):
  move=state.p1.active_pokemon.moves[0].clone();move.base_power=power
  expected=set(calculate_damage_rolls(state.p1.active_pokemon,state.p2.active_pokemon,move))
  observed={r['damageTrace'][hit]['damage'] for r in rows if len(r['damageTrace'])>hit}
  assert observed==expected==local[hit]


@pytest.mark.parametrize('move,weather,ability,expected',[
 ('Hurricane','raindance','Shell Armor',1.),('Thunder','raindance','Shell Armor',1.),
 ('Hurricane','sunnyday','Shell Armor',.5),('Blizzard','snowscape','Shell Armor',1.),
 ('Hurricane','raindance','Cloud Nine',.7)])
def test_weather_accuracy(move,weather,ability,expected):
 rows=run([dict(name=str(i),weather=weather,teams=[[team('Mew',moves=[move])],[team(ability=ability)]],seed=[i+1,2,3,4]) for i in range(512)])
 counts=[0,0]
 for i,row in enumerate(rows):
  state=from_snapshot(row['before'])
  after=simulate_turn_transition(state,MoveAction(state.p1.active_pokemon.moves[0].id,1),MoveAction('splash',1),sample_outcomes=True,rng=random.Random(i))
  counts[0]+=any(d['side']==2 and d['effect']==state.p1.active_pokemon.moves[0].id for d in row['damageTrace'])
  counts[1]+=after.p2.active_pokemon.current_hp<state.p2.active_pokemon.current_hp
 assert all(abs(n/512-expected)<.08 for n in counts)
 if expected==1.:assert counts==[512,512]
