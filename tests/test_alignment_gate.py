import hashlib,json
from pathlib import Path
import pytest
from glaubermon.evaluation.alignment_gate import require_pilot_alignment
from glaubermon.models.embeddings import FEATURE_SCHEMA


def fixture(tmp_path):
    (tmp_path/'docs').mkdir();(tmp_path/'showdown').mkdir()
    (tmp_path/'showdown/package.json').write_text(json.dumps({'version':'0.11.11'}))
    (tmp_path/'engine.py').write_text('verified implementation')
    pool={'practice':'packed'}
    scope={'teams':{'practice':{'packed':'packed','sha256':hashlib.sha256(b'packed').hexdigest()}}}
    (tmp_path/'docs/scope.json').write_text(json.dumps(scope))
    status=dict(training_allowed=True,feature_schema=FEATURE_SCHEMA,pilot_scope='docs/scope.json',pilot_scope_sha256=hashlib.sha256((tmp_path/'docs/scope.json').read_bytes()).hexdigest(),official_reference='pokemon-showdown@0.11.11',validated_files={'engine.py':hashlib.sha256((tmp_path/'engine.py').read_bytes()).hexdigest()})
    (tmp_path/'docs/ALIGNMENT_STATUS.json').write_text(json.dumps(status))
    return pool


def test_clearance_accepts_verified_official_pool(tmp_path):
    pool=fixture(tmp_path)
    assert require_pilot_alignment('showdown',tmp_path/'showdown',tmp_path,pool)['training_allowed']

@pytest.mark.parametrize('change',['internal','team','version','implementation','scope'])
def test_clearance_rejects_drift_before_training(tmp_path,change):
    pool=fixture(tmp_path);backend='showdown'
    if change=='internal':backend='internal-privileged'
    if change=='team':pool['practice']='other moves'
    if change=='version':(tmp_path/'showdown/package.json').write_text('{"version":"changed"}')
    if change=='implementation':(tmp_path/'engine.py').write_text('unverified implementation')
    if change=='scope':(tmp_path/'docs/scope.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Training paused'):
        require_pilot_alignment(backend,tmp_path/'showdown',tmp_path,pool)
