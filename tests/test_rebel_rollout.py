"""Exercise a neural self-play trajectory and a real optimizer update."""
import json
import math
import torch
import pytest

from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Move, Pokemon
from glaubermon.core.types import Hazard, Terrain
from glaubermon.scripts import train_rebel


def make_mon(hp=1):
    return Pokemon(species="Snorlax", ability="Immunity", max_hp=461, current_hp=hp,
                   raw_stats={"hp":461,"atk":256,"def":166,"spa":166,"spd":256,"spe":96},
                   moves=[Move.from_dex("tackle")])


def test_neural_self_play_then_optimizer_step(tmp_path, monkeypatch):
    monkeypatch.setattr(train_rebel.signal, "signal", lambda *args: None)
    monkeypatch.setattr(train_rebel, "generate_competitive_battle",
                        lambda: BattleState(BattleSide([make_mon()]), BattleSide([make_mon()])))
    trainer = train_rebel.AlphaZeroTrainer(checkpoint_dir=str(tmp_path), d_model=32,
                                         nhead=4, reset_from_scratch=True, rollout_backend="internal-privileged")
    calls = []
    hook = trainer.model.register_forward_hook(lambda *args: calls.append(1))
    trainer.model.train()  # reproduce state left by the previous optimizer step
    samples = trainer.play_self_play_game()
    hook.remove()
    assert calls and not trainer.model.training
    assert len(samples) == 1
    assert samples[0][2] in (-1.0, 1.0)
    assert abs(samples[0][1].sum()-1) < 1e-6
    for sample in samples:
        trainer.replay_buffer.push(*sample)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    losses = trainer.train_step(batch_size=1)
    assert all(math.isfinite(v) for v in losses)
    assert any(not torch.equal(old,new) for old,new in zip(before,trainer.model.parameters()))
    trainer.save_checkpoint()
    assert json.loads((tmp_path/'rebel_meta.json').read_text())['rollout_mode'] == 'sampled_internal_privileged_v2'


def test_replacement_phase_handles_chain_of_hazard_kos(monkeypatch):
    state = BattleState(BattleSide([make_mon(0),make_mon(1),make_mon(461)],
                                  hazards={Hazard.STEALTH_ROCK:1}),
                        BattleSide([make_mon(461)]), terrain=Terrain.GRASSY, trick_room=3)
    def choose_first(resolver, side, opponent, view):
        assert view.trick_room == 3 and view.terrain == Terrain.GRASSY
        return side.available_switches()[0]
    monkeypatch.setattr(train_rebel, "execute_showdown_accurate_force_switch", choose_first)
    trainer = object.__new__(train_rebel.AlphaZeroTrainer)
    trainer._replace_fainted(state, None)
    assert state.p1.active_index == 2
    assert state.p1.pokemon[1].is_fainted
    assert state.p1.active_pokemon.current_hp == 461-57


def test_incompatible_checkpoint_fails_instead_of_training_random_weights(tmp_path):
    torch.save({}, tmp_path/'glaubermon_rebel_latest.pt')
    with pytest.raises(RuntimeError, match="Missing key"):
        train_rebel.AlphaZeroTrainer(checkpoint_dir=str(tmp_path), d_model=32, nhead=4, rollout_backend="internal-privileged")


def test_truncation_has_no_value_target_or_value_head_update(tmp_path, monkeypatch):
    monkeypatch.setattr(train_rebel.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(train_rebel, 'generate_competitive_battle',
                        lambda: BattleState(BattleSide([make_mon(461)]),BattleSide([make_mon(461)])))
    trainer = train_rebel.AlphaZeroTrainer(checkpoint_dir=str(tmp_path), d_model=32,
                    nhead=4, reset_from_scratch=True, rollout_backend='internal-privileged', max_turns=1)
    samples = trainer.play_self_play_game()
    assert samples and all(sample[2] is None for sample in samples)
    for sample in samples:
        trainer.replay_buffer.push(*sample)
    value_before = [p.detach().clone() for p in trainer.model.value_head.parameters()]
    policy_before = [p.detach().clone() for p in trainer.model.policy_head.parameters()]
    v_loss,p_loss = trainer.train_step(batch_size=1)
    assert v_loss == 0 and math.isfinite(p_loss)
    assert all(torch.equal(a,b) for a,b in zip(value_before,trainer.model.value_head.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(policy_before,trainer.model.policy_head.parameters()))


def test_official_rollout_uses_both_private_views_and_marks_cap(tmp_path, monkeypatch):
    from pathlib import Path
    showdown = Path(__file__).resolve().parents[2] / 'showdown-parity/node_modules/pokemon-showdown'
    if not showdown.exists():
        showdown = Path(__file__).resolve().parents[1] / 'tools/showdown/node_modules/pokemon-showdown'
    if not showdown.exists():
        pytest.skip('Official simulator unavailable')
    monkeypatch.setattr(train_rebel.signal, 'signal', lambda *args: None)
    trainer = train_rebel.AlphaZeroTrainer(checkpoint_dir=str(tmp_path), d_model=32,
                      nhead=4, reset_from_scratch=True, max_turns=2, mechanics_seed=19,
                      showdown_path=str(showdown))
    samples = trainer.play_self_play_game()
    assert len(samples) >= 2
    assert not trainer.last_game_info['terminated']
    assert all(sample[2] is None for sample in samples)
    assert all(abs(sample[1].sum()-1)<1e-6 for sample in samples)
    # P1 and P2 actions both become examples with their own side first.
    trace = [json.loads(l) for l in (tmp_path/'rollouts/game-000000.jsonl').read_text().splitlines()]
    requests = [r['frame']['players'] for r in trace if 'frame' in r]
    assert requests[0][0]['request']['side']['name'] == 'SelfPlay1'
    assert requests[0][1]['request']['side']['name'] == 'SelfPlay2'
    assert trainer.rollout_mode == 'official_public_v2'


def test_resume_rejects_changed_training_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(train_rebel.signal,'signal',lambda *args:None)
    trainer=train_rebel.AlphaZeroTrainer(checkpoint_dir=str(tmp_path),d_model=32,nhead=4,
                        reset_from_scratch=True,rollout_backend='internal-privileged',max_turns=1)
    trainer.save_checkpoint()
    with pytest.raises(ValueError,match='contract changed'):
        train_rebel.AlphaZeroTrainer(checkpoint_dir=str(tmp_path),d_model=32,nhead=4,
                                    rollout_backend='internal-privileged',max_turns=2)
