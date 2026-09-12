"""Every pinned move is exercised with its actual user, item, ability and stats."""
import pytest
from test_pilot_trajectories import SCOPE
from test_paradox_reference import run
from test_volatile_phases_reference import test_volatile_and_phase_trajectory as _compare

def simple(species,move):return '|'.join([species,'','','shellarmor',move,'Hardy','','','','','',''])

CASES=[]
for move in SCOPE['inventory']['moves']:
 packed=next(mon for team in SCOPE['teams'].values() for mon in team['packed'].split(']') if move in mon.split('|')[4].split(','))
 slot=packed.split('|')[4].split(',').index(move)+1
 initial=[{'hp':100},{}]
 if move=='sleeptalk':initial[0].update(status='slp',time=3)
 if move=='defog':
  initial[0]['hazards']=['stealthrock'];initial[1].update(hazards=['spikes'],screens=['reflect'])
 CASES.append(dict(name='inventory_'+move,packed_teams=[packed+']'+simple('Blissey','splash'), simple('Mew','tackle')+']'+simple('Blissey','splash')],initial=initial,
                   controlled_draws=True,actions=[[f'move {slot}','move 1']],compare_types=True,compare_ability=True))

@pytest.fixture(scope='module')
def reference():return {r['name']:r for r in run(CASES)}

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_every_pilot_move_transition(case,reference):_compare(case,reference)
