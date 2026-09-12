"""Whole five-team trajectories with explicitly coupled semantic random outcomes."""
import json
from pathlib import Path
import pytest
from test_paradox_reference import run
from test_volatile_phases_reference import test_volatile_and_phase_trajectory as _compare

SCOPE=json.loads((Path(__file__).resolve().parents[1]/'docs/alignment/pilot-scope.json').read_text())
NAMES=['development_offense','development_rain','balance','stall','pelol']
CASES=[dict(name=f'{a}_vs_{b}',packed_teams=[SCOPE['teams'][name]['packed'] for name in (a,b)],controlled_draws=True,auto_steps=1200,compare_types=True,compare_ability=True,compare_legal=True) for a,b in zip(NAMES,NAMES[1:]+NAMES[:1])]

@pytest.fixture(scope='module')
def reference():return {r['name']:r for r in run(CASES)}

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_whole_pilot_trajectory(case,reference):
 assert reference[case['name']]['ended'], 'Controlled scenario must reach a real terminal state'
 _compare(case,reference)
