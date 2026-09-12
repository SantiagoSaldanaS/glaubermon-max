"""Current-team move failures and item/absorption callbacks against Showdown."""
import pytest
from test_paradox_reference import run
from test_alignment_reference import from_snapshot
from test_field_multihit_reference import team
from test_volatile_phases_reference import test_volatile_and_phase_trajectory as _compare
from glaubermon.inference.damage_calc import calculate_damage_rolls

CASES=[]
for move in ('Thunderclap','Sucker Punch'):
 CASES.extend([
 dict(name=move+'_fails_status',teams=[[team('Mew',moves=[move])],[team()]]),
 dict(name=move+'_fails_switch',teams=[[team('Mew',moves=[move])],[team(),team('Blissey')]],actions=[['move 1','switch 2']]),
 dict(name=move+'_fails_after_priority',teams=[[team('Mew',moves=[move])],[team(moves=['Extreme Speed'])]],initial=[{}, {'stats':{'atk':1}}]),
 dict(name=move+'_hits_pending_attack',teams=[[team('Mew',moves=[move])],[team(moves=['Seismic Toss'])]],initial=[{}, {'hp':1}]),
 ])
CASES += [
 dict(name='water_absorb_heals',teams=[[team(moves=['Surf'])],[team('Ogerpon-Wellspring',ability='Water Absorb',moves=['Splash'])]],initial=[{}, {'hp':100}]),
 dict(name='water_absorb_full_hp',teams=[[team(moves=['Surf'])],[team(ability='Water Absorb')]]),
 dict(name='water_absorb_precedes_substitute',teams=[[team(moves=['Surf'])],[team(ability='Water Absorb')]],initial=[{}, {'hp':100,'volatiles':{'substitute':{'hp':100}}}]),
 dict(name='water_absorb_before_accuracy',teams=[[team(moves=['Hydro Pump'])],[team(ability='Water Absorb')]],initial=[{'boosts':{'accuracy':-6}}, {'hp':100}]),
 dict(name='water_absorb_stops_flip_turn',teams=[[team('Mew',moves=['Flip Turn']),team('Blissey')],[team(ability='Water Absorb')]],initial=[{}, {'hp':100}]),
 dict(name='flash_fire_status_absorption',teams=[[team(moves=['Will-O-Wisp'])],[team('Mew',ability='Flash Fire')]]),
 dict(name='flash_fire_absorbs_before_accuracy',teams=[[team(moves=['Fire Blast'])],[team('Heatran',ability='Flash Fire')]],initial=[{'boosts':{'accuracy':-6}},{}]),
 dict(name='flash_fire_blocks_trapping',teams=[[team(moves=['Magma Storm'])],[team('Heatran',ability='Flash Fire')]]),
 dict(name='flash_fire_through_substitute',teams=[[team(moves=['Flamethrower'])],[team('Heatran',ability='Flash Fire')]],initial=[{}, {'volatiles':{'substitute':{'hp':50}}}]),
 dict(name='flash_fire_clears_on_switch',teams=[[team(moves=['Will-O-Wisp','Splash'])],[team('Heatran',ability='Flash Fire'),team('Blissey')]],actions=[['move 1','move 1'],['move 2','switch 2'],['move 2','switch 2']]),
 dict(name='protect_prevents_absorption',teams=[[team(moves=['Surf'])],[team(ability='Water Absorb',moves=['Protect'])]],initial=[{}, {'hp':100}]),
 dict(name='lum_cures_toxic_before_residual',teams=[[team('Mew',moves=['Toxic'])],[team(item='Lum Berry')]]),
 dict(name='lum_cures_burn_before_attack',teams=[[team('Mew',moves=['Will-O-Wisp'])],[team(item='Lum Berry',moves=['Seismic Toss'])]]),
 dict(name='lum_cures_confusion',teams=[[team('Mew',moves=['Confuse Ray'])],[team(item='Lum Berry')]]),
 dict(name='lum_cures_rest',teams=[[team('Mew',item='Lum Berry',moves=['Rest'])],[team()]],initial=[{'hp':100},{}]),
 dict(name='lum_cures_nuzzle_before_action',teams=[[team('Mew',moves=['Nuzzle'])],[team(item='Lum Berry',moves=['Seismic Toss'])]],initial=[{'stats':{'atk':1}},{}]),
 dict(name='lum_not_consumed_through_substitute',teams=[[team('Mew',moves=['Nuzzle'])],[team(item='Lum Berry')]],initial=[{'stats':{'atk':1}},{'volatiles':{'substitute':{'hp':100}}}]),
 dict(name='lum_cures_toxic_spikes_entry',teams=[[team(),team('Blissey',item='Lum Berry')],[team()]],initial=[{'hazards':['toxicspikes']},{}],actions=[['switch 2','move 1']]),
 dict(name='lum_not_consumed_on_ko',teams=[[team('Mew',moves=['Nuzzle'])],[team(item='Lum Berry')]],initial=[{}, {'hp':1}]),
]

TYPE_CASES = [
 dict(name='protean_changes_before_protect',teams=[[team('Meowscarada',ability='Protean',moves=['Triple Axel'])],[team(moves=['Protect'])]]),
 dict(name='protean_once_per_entry',teams=[[team('Meowscarada',ability='Protean',moves=['Splash','Swords Dance','Agility'])],[team()]],actions=[['move 1','move 1'],['move 2','move 1'],['move 3','move 1']]),
 dict(name='protean_same_type_does_not_consume_activation',teams=[[team('Smeargle',ability='Protean',moves=['Splash','Agility'])],[team()]],actions=[['move 1','move 1'],['move 2','move 1']]),
 dict(name='protean_resets_on_switch',teams=[[team('Meowscarada',ability='Protean',moves=['Splash','Agility']),team('Blissey')],[team()]],actions=[['move 1','move 1'],['switch 2','move 1'],['switch 2','move 1'],['move 2','move 1']]),
 dict(name='protean_not_used_on_failed_thunderclap',teams=[[team('Meowscarada',ability='Protean',moves=['Thunderclap'])],[team()]]),
 dict(name='protean_cannot_change_tera',teams=[[team('Meowscarada',ability='Protean',moves=['Agility'],teraType='Grass')],[team()]],actions=[['move 1 terastallize','move 1']]),
 dict(name='roost_ends_at_residual',teams=[[team('Corviknight',moves=['Roost'])],[team()]],initial=[{'hp':100},{}]),
 dict(name='roost_fails_at_full_hp',teams=[[team('Corviknight',moves=['Roost'])],[team()]]),
 dict(name='roost_exposes_to_ground',teams=[[team('Corviknight',moves=['Roost'],evs={'spe':252})],[team(moves=['Earthquake'])]],initial=[{'hp':100},{'stats':{'atk':1}}]),
 dict(name='roost_pure_flying_becomes_normal',teams=[[team('Tornadus',moves=['Roost'])],[team(moves=['Earthquake'])]],initial=[{'hp':100},{'stats':{'atk':1}}]),
 dict(name='roost_flying_tera_stays_immune',teams=[[team('Corviknight',moves=['Roost'],teraType='Flying')],[team(moves=['Earthquake'])]],initial=[{'hp':100},{}],actions=[['move 1 terastallize','move 1']]),
 dict(name='roost_preserves_type_across_pivot_phase',teams=[[team('Corviknight',moves=['Roost'],evs={'spe':252})],[team(moves=['U-turn']),team('Blissey')]],initial=[{'hp':100},{'stats':{'atk':1}}],actions=[['move 1','move 1'],['','switch 2']]),
 dict(name='protean_struggle_does_not_change_type',teams=[[team('Meowscarada',ability='Protean',moves=['Tackle'])],[team()]],initial=[{'pp':[0]}, {'hp':1}]),
]
for c in TYPE_CASES:c['compare_types']=True
CASES.extend(TYPE_CASES)

@pytest.fixture(scope='module')
def reference():return {r['name']:r for r in run(CASES)}

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_team_callback_trajectory(case,reference):_compare(case,reference)

@pytest.mark.parametrize('move',['Fire Punch','Flamethrower'])
def test_flash_fire_activated_damage_support(move):
 rows=run([dict(name=str(i),teams=[[team('Heatran',ability='Flash Fire',moves=[move])],[team()]],initial=[{'volatiles':{'flashfire':{}}},{}],seed=[i+1,2,3,4]) for i in range(256)])
 state=from_snapshot(rows[0]['before']);a,d=state.p1.active_pokemon,state.p2.active_pokemon
 actual={hit['damage'] for row in rows for hit in row['damageTrace'] if hit['side']==2 and hit['effect']==a.moves[0].id and not hit['critical']}
 assert actual==set(calculate_damage_rolls(a,d,a.moves[0]))


@pytest.mark.parametrize('used',[False,True])
def test_protean_damage_query_does_not_mutate_attacker(used):
 cases=[dict(name=str(i),teams=[[team('Meowscarada',ability='Protean',moves=['Ice Punch'])],[team()]],seed=[i+1,2,3,4]) for i in range(256)]
 # A prior Splash activates Protean to Normal before the attack; only the first
 # attack can change type on a fresh entry.
 if used:
  for case in cases:
   case['teams'][0][0]['moves']=['Splash','Ice Punch']
   case['actions']=[['move 1','move 1'],['move 2','move 1']]
 rows=run(cases);state=from_snapshot(rows[0]['results'][0] if used else rows[0]['before'])
 a,d=state.p1.active_pokemon,state.p2.active_pokemon;move=a.moves[-1]
 original=(a.types,a.type_override,a.protean_used)
 actual={hit['damage'] for row in rows for hit in row['damageTrace'] if hit['side']==2 and hit['effect']=='icepunch' and not hit['critical']}
 assert actual==set(calculate_damage_rolls(a,d,move))
 assert (a.types,a.type_override,a.protean_used)==original
