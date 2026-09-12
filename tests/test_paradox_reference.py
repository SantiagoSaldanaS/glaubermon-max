"""Paradox activation lifecycle and damage, compared with Gen 9 Showdown."""
import json,subprocess
import pytest
from test_alignment_reference import ROOT,SHOWDOWN,from_snapshot
from test_field_multihit_reference import team
from test_volatile_phases_reference import test_volatile_and_phase_trajectory as _compare
from glaubermon.inference.damage_calc import calculate_damage_rolls

CASES=[]
for species,ability,field,start,stop in [('Great Tusk','Protosynthesis',{'weather':'sunnyday'},'Sunny Day','Rain Dance'),('Iron Treads','Quark Drive',{'terrain':'electricterrain'},'Electric Terrain','Misty Terrain')]:
 CASES.extend([
 dict(name=ability+'_agility_does_not_retarget_speed_boost',**field,teams=[[team(species,moves=['Agility','Seismic Toss'],ability=ability)],[team(moves=['Splash','Seismic Toss'])]],initial=[{'hp':100,'stats':{'atk':201,'def':100,'spa':100,'spd':100,'spe':200}},{'hp':100,'stats':{'spe':500}}],actions=[['move 1','move 1'],['move 2','move 2']]),
 dict(name=ability+'_entry_energy',teams=[[team('Mew'),team(species,ability=ability,item='Booster Energy')],[team()]],actions=[['switch 2','move 1']]),
 dict(name=ability+'_entry_saves_energy_then_consumes',**field,teams=[[team('Mew'),team(species,ability=ability,item='Booster Energy')],[team(moves=['Splash',stop])]],actions=[['switch 2','move 1'],['move 1','move 2']]),
 dict(name=ability+'_stat_locked_despite_agility',**field,teams=[[team(species,moves=['Agility','Splash'],ability=ability)],[team()]],initial=[{'stats':{'atk':201,'def':100,'spa':100,'spd':100,'spe':200}},{}],actions=[['move 1','move 1'],['move 2','move 1']]),
 dict(name=ability+'_energy_persists_when_field_starts',teams=[[team(species,ability=ability,item='Booster Energy')],[team(moves=[start])]]),
 dict(name=ability+'_field_ends_without_item',**field,teams=[[team(species,ability=ability)],[team(moves=[stop])]]),
 dict(name=ability+'_boost_ends_on_switch',teams=[[team(species,ability=ability,item='Booster Energy'),team('Mew')],[team()]],actions=[['switch 2','move 1'],['switch 2','move 1']]),
 ])
CASES += [
 dict(name='sun_expiry_reactivates_energy',weather='sunnyday',weather_turns=2,teams=[[team('Mew'),team('Great Tusk',ability='Protosynthesis',item='Booster Energy')],[team()]],actions=[['switch 2','move 1'],['move 1','move 1']]),
 dict(name='cloud_nine_ends_solar_boost',weather='sunnyday',teams=[[team('Great Tusk',ability='Protosynthesis')],[team(),team('Golduck',ability='Cloud Nine')]],actions=[['move 1','switch 2']]),
 dict(name='ice_spinner_ends_terrain_boost',terrain='electricterrain',teams=[[team('Iron Treads',ability='Quark Drive')],[team('Mew',moves=['Ice Spinner'])]],initial=[{}, {'stats':{'atk':1}}]),
]


def run(cases):
 if not SHOWDOWN.exists():pytest.skip('Install pinned official simulator')
 p=subprocess.run(['node',str(ROOT/'tests/fixtures/alignment_reference.cjs'),str(SHOWDOWN)],input=json.dumps(cases),capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 return json.loads(p.stdout)

@pytest.fixture(scope='module')
def reference():return {r['name']:r for r in run(CASES)}

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_activation_trajectory(case,reference):_compare(case,reference)

@pytest.mark.parametrize('ability,field',[('Protosynthesis',{'weather':'sunnyday'}),('Quark Drive',{'terrain':'electricterrain'})])
@pytest.mark.parametrize('stat,move,defending',[('atk','Tackle',False),('spa','Swift',False),('def','Tackle',True),('spd','Swift',True)])
def test_paradox_damage_support(ability,field,stat,move,defending):
 index=int(defending);teams=[[team('Mew',moves=[move])],[team()]]
 teams[index][0]['ability']=ability
 initial=[{},{}];initial[index]={'stats':{k:(201 if k==stat else 100) for k in ('atk','def','spa','spd','spe')}}
 rows=run([dict(name=str(i),**field,teams=teams,initial=initial,seed=[i+1,2,3,4]) for i in range(256)])
 state=from_snapshot(rows[0]['before']);a,d=state.p1.active_pokemon,state.p2.active_pokemon
 # Defensive Paradox replaces Shell Armor; exclude critical observations explicitly.
 rolls=calculate_damage_rolls(a,d,a.moves[0],state.weather,state.terrain)
 actual={hit['damage'] for row in rows for hit in row['damageTrace'] if hit['side']==2 and hit['effect']==a.moves[0].id and not hit['critical']}
 assert actual==set(rolls)


@pytest.mark.parametrize('ability,item,defender_ability,boosts,critical',[
 ('Shell Armor','','Shell Armor',{'def':2,'atk':6},False),
 ('Fur Coat','','Shell Armor',{'def':1},False),
 ('Shell Armor','Eviolite','Shell Armor',{'def':1},False),
 ('Shell Armor','Choice Band','Shell Armor',{'def':1},False),
 ('Shell Armor','','Unaware',{'def':2},False),
 ('Shell Armor','','Thick Fat',{'def':-2,'atk':6},True),
])
def test_body_press_uses_defense_with_attack_modifiers(ability,item,defender_ability,boosts,critical):
 rows=run([dict(name=str(i),teams=[[team('Chansey' if item=='Eviolite' else 'Mew',moves=['Body Press'],ability=ability,item=item)],[team('Mew',ability=defender_ability)]],initial=[{'boosts':boosts,'stats':{'atk':99,'def':161}},{}],seed=[i+1,2,3,4]) for i in range(512 if critical else 256)])
 state=from_snapshot(rows[0]['before']);a,d=state.p1.active_pokemon,state.p2.active_pokemon
 rolls=calculate_damage_rolls(a,d,a.moves[0],is_critical=critical)
 actual={hit['damage'] for row in rows for hit in row['damageTrace'] if hit['side']==2 and hit['effect']=='bodypress' and hit['critical']==critical}
 assert actual <= set(rolls)
 if not critical:assert actual==set(rolls)
 assert actual
