"""Regressions discovered while replaying complete pilot-team trajectories."""
import pytest
from test_field_multihit_reference import team
from test_paradox_reference import run
from test_alignment_reference import from_snapshot
from test_volatile_phases_reference import test_volatile_and_phase_trajectory as _compare
from glaubermon.inference.damage_calc import calculate_damage_rolls

CASES=[
 dict(name='struggle_menu_excludes_tera',teams=[[team('Mew',moves=['Tackle'],teraType='Grass')],[team()]],initial=[{'pp':[0]},{}],actions=[['move 1','move 1']]),
 dict(name='queued_attack_without_target_still_spends_pp',teams=[[team('Mew',ability='Pressure',moves=['Tackle']),team('Blissey')],[team(moves=['Seismic Toss'])]],initial=[{'hp':1,'pp':[0]},{}]),
 dict(name='self_recovery_after_recoil_ko_still_runs',teams=[[team('Mew',moves=['Tackle']),team('Blissey')],[team(moves=['Recover'])]],initial=[{'hp':1,'pp':[0]},{'hp':100}]),
 dict(name='fainted_tera_clears_type_but_side_cannot_reuse',teams=[[team('Mew',moves=['Splash'],teraType='Grass'),team('Blissey')],[team(moves=['Extreme Speed'])]],initial=[{'hp':1},{}],actions=[['move 1 terastallize','move 1'],['switch 2','']]),
]
for hp in (2,5,8):
 CASES.append(dict(name=f'brave_bird_rounds_clipped_recoil_{hp}',teams=[[team('Corviknight',moves=['Brave Bird'])],[team(),team('Blissey')]],initial=[{'hp':100},{'hp':hp}]))
for ability in ('Shell Armor','Infiltrator'):
 CASES.append(dict(name='defog_substitute_'+ability,teams=[[team('Corviknight',ability=ability,moves=['Defog'])],[team()]],initial=[{'screens':['reflect'],'hazards':['stealthrock']},{'screens':['lightscreen'],'hazards':['spikes'],'volatiles':{'substitute':{'hp':40}}}]))
CASES.append(dict(name='reflected_defog_clears_source_screens',teams=[[team('Corviknight',moves=['Defog'])],[team(ability='Magic Bounce')]],initial=[{'screens':['reflect'],'hazards':['stealthrock']},{'screens':['lightscreen'],'hazards':['spikes']}]))
for c in CASES:c.update(controlled_draws=True,compare_types=True)

@pytest.fixture(scope='module')
def reference():return {r['name']:r for r in run(CASES)}

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_whole_trajectory_regression(case,reference):_compare(case,reference)

@pytest.mark.parametrize('ability',['Synchronize','Shell Armor'])
def test_guaranteed_critical_in_damage_query(ability):
 rows=run([dict(name=str(i),teams=[[team('Meowscarada',ability='Protean',moves=['Flower Trick'])],[team('Mew',ability=ability)]],initial=[{'boosts':{'atk':-2}},{'boosts':{'def':4},'screens':['reflect']}],seed=[i+1,2,3,4]) for i in range(256)])
 state=from_snapshot(rows[0]['before']);a,d=state.p1.active_pokemon,state.p2.active_pokemon
 actual={hit['damage'] for row in rows for hit in row['damageTrace'] if hit['side']==2 and hit['effect']=='flowertrick'}
 assert actual==set(calculate_damage_rolls(a,d,a.moves[0],defender_side=state.p2))
 assert {hit['critical'] for row in rows for hit in row['damageTrace'] if hit['effect']=='flowertrick'}=={ability!='Shell Armor'}
