"""Limit alignment clearance to the verified official training route and team pool."""
import hashlib
import json
from pathlib import Path
from glaubermon.models.embeddings import FEATURE_SCHEMA
from glaubermon.evaluation.training_teams import TRAINING_TEAMS

ROOT=Path(__file__).resolve().parents[2]


def require_pilot_alignment(backend,showdown_path,root=ROOT,team_pool=None):
    status_path=root/'docs/ALIGNMENT_STATUS.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    def paused(reason):raise RuntimeError('Training paused: '+reason)
    if not status.get('training_allowed'):paused('pending environment alignment; see docs/ALIGNMENT_STATUS.json')
    if backend!='showdown':paused('pilot clearance requires official Showdown rollouts')
    if status.get('feature_schema')!=FEATURE_SCHEMA:paused('observation schema changed')
    scope_path=root/status['pilot_scope']
    if hashlib.sha256(scope_path.read_bytes()).hexdigest()!=status.get('pilot_scope_sha256'):paused('pilot scope changed')
    scope=json.loads(scope_path.read_text())
    pool=TRAINING_TEAMS if team_pool is None else team_pool
    if not pool:paused('training team pool is empty')
    for name,packed in pool.items():
        expected=scope['teams'].get(name,{})
        if expected.get('packed')!=packed or expected.get('sha256')!=hashlib.sha256(packed.encode()).hexdigest():
            paused('unverified training team: '+name)
    package=Path(showdown_path)/'package.json'
    if not package.exists() or 'pokemon-showdown@'+json.loads(package.read_text())['version']!=status.get('official_reference'):
        paused('official simulator version changed')
    files=status.get('validated_files',{})
    if not files:paused('missing validation fingerprints')
    for name,digest in files.items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:paused('validated implementation changed: '+name)
    return status
