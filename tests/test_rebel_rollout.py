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
                                         nhead=4, reset_from_scratch=True)
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
    assert json.loads((tmp_path/'rebel_meta.json').read_text())['rollout_mode'] == 'sampled_internal_v1_experimental'


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
        train_rebel.AlphaZeroTrainer(checkpoint_dir=str(tmp_path), d_model=32, nhead=4)
