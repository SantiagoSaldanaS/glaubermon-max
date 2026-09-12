"""Reference integration: validation, reproducibility, and private channel separation."""
from pathlib import Path
import pytest
from glaubermon.evaluation.official_benchmark import OfficialBridge, choice_text
from glaubermon.client.showdown_bot import PACKED_TEAMS

SHOWDOWN = Path(__file__).resolve().parents[2] / 'showdown-parity/node_modules/pokemon-showdown'
if not SHOWDOWN.exists():
    SHOWDOWN = Path(__file__).resolve().parents[1] / 'tools/showdown/node_modules/pokemon-showdown'
pytestmark = pytest.mark.skipif(not SHOWDOWN.exists(), reason='pinned official npm simulator not installed')


def start(bridge, team='balance'):
    return bridge.exchange(dict(op='start', teams=[PACKED_TEAMS[team], PACKED_TEAMS['stall']],
                                names=['Glaubermon','Control'], seed=[1,911,23456,34567]))


def test_official_seed_reproducibility_and_private_channels():
    bridge = OfficialBridge(SHOWDOWN)
    try:
        initial = start(bridge)
        assert initial['players'][0]['request']['side']['name'] == 'Glaubermon'
        assert initial['players'][1]['request']['side']['name'] == 'Control'
        first = bridge.exchange(dict(op='choose', choices=['team 123456','team 123456']))
        start(bridge)
        second = bridge.exchange(dict(op='choose', choices=['team 123456','team 123456']))
        assert first == second
        for side in (0,1):
            channel = first['players'][side]
            assert channel['request']['side']['id'] == f'p{side+1}'
            own = [l for l in channel['lines'] if l.startswith(f'|switch|p{side+1}')]
            other = [l for l in channel['lines'] if l.startswith(f'|switch|p{2-side}')]
            assert own and other
            assert '/100' not in own[0].split('|')[-1]
            assert '/100' in other[0].split('|')[-1]
            assert not any('|split|' in l for l in channel['lines'])
    finally:
        bridge.close()


def test_illegal_team_is_rejected_before_start():
    bridge = OfficialBridge(SHOWDOWN)
    try:
        with pytest.raises(RuntimeError, match='Roaring Moon'):
            start(bridge, 'hyper_offense')
    finally:
        bridge.close()


def test_adapter_preserves_official_choice_syntax():
    assert choice_text('battle-local|/choose move 2 terastallize') == 'move 2 terastallize'
    assert choice_text('/choose switch Great Tusk') == 'switch Great Tusk'
    assert choice_text('/team 654321') == 'team 654321'
