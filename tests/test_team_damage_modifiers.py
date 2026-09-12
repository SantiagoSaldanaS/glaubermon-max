"""Exact damage supports for modifier combinations occurring in the pilot pool."""
import pytest
from test_paradox_reference import run
from test_alignment_reference import from_snapshot
from test_field_multihit_reference import team
from glaubermon.inference.damage_calc import calculate_damage_rolls

CASES=[]
for move in ('Ivy Cudgel','Horn Leech','Play Rough'):
 CASES.append(dict(name='mask_'+move,teams=[[team('Ogerpon-Wellspring',item='Wellspring Mask',ability='Water Absorb',moves=[move])],[team()]]))
for fallen in range(6):
 CASES.append(dict(name='overlord_black_glasses_'+str(fallen),teams=[[team('Kingambit',ability='Supreme Overlord',item='Black Glasses',moves=['Kowtow Cleave'])],[team()]],initial=[{'fallen':fallen},{}]))
for item in ('','Choice Specs','Life Orb'):
 for ability,weather in (('Shell Armor',''),('Protosynthesis','sunnyday')):
  CASES.append(dict(name='vessel_'+item+'_'+ability,weather=weather,teams=[[team('Mew',ability=ability,item=item,moves=['Shadow Ball'])],[team('Ting-Lu',ability='Vessel of Ruin')]],initial=[{'stats':{'spa':321}},{}]))
for screen in (False,True):
 for ability in ('Shell Armor','Multiscale'):
  CASES.append(dict(name='life_orb_'+str(screen)+'_'+ability,teams=[[team('Mew',item='Life Orb',moves=['Moonblast'])],[team('Dragonite',ability=ability)]],initial=[{}, {'screens':['lightscreen'] if screen else []}]))
CASES += [
 dict(name='choice_band_body_press',teams=[[team('Zamazenta',item='Choice Band',moves=['Body Press'])],[team('Mew')]],initial=[{'stats':{'def':301},'boosts':{'def':1}},{'stats':{'def':301}}]),
 dict(name='quark_life_orb',terrain='electricterrain',teams=[[team('Iron Valiant',ability='Quark Drive',item='Life Orb',moves=['Moonblast'])],[team()]]),
 dict(name='flash_fire_specs',teams=[[team('Heatran',ability='Flash Fire',item='Choice Specs',moves=['Flamethrower'])],[team()]],initial=[{'stats':{'spa':301},'volatiles':{'flashfire':{}}},{}]),
]

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_current_team_damage_support(case):
 rows=run([dict(case,name=str(i),seed=[i+1,2,3,4]) for i in range(256)])
 state=from_snapshot(rows[0]['before']);a,d=state.p1.active_pokemon,state.p2.active_pokemon
 actual={hit['damage'] for row in rows for hit in row['damageTrace'] if hit['side']==2 and hit['effect']==a.moves[0].id and not hit['critical']}
 predicted=set(calculate_damage_rolls(a,d,a.moves[0],state.weather,state.terrain,attacker_side=state.p1,defender_side=state.p2))
 assert actual==predicted,(case['name'],sorted(actual),sorted(predicted))
