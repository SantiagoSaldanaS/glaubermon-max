"""Regression cases for rollout randomness, mutable state and PP accounting."""
import random
from collections import Counter

from glaubermon.core.actions import MoveAction
from glaubermon.core.battle_state import BattleSide, BattleState
from glaubermon.core.pokemon import Move, Pokemon
from glaubermon.core.types import PokemonType, StatusCondition, MoveCategory
from glaubermon.inference.damage_calc import calculate_damage_rolls
from glaubermon.search.subgame_resolver import simulate_turn_transition


def battle(first="tackle", second="amnesia", hp=None):
    def mon(move):
        return Pokemon(species="Snorlax", ability="Immunity", max_hp=461,
                       current_hp=461 if hp is None else hp,
                       raw_stats={"hp":461,"atk":256,"def":166,"spa":166,"spd":256,"spe":96},
                       moves=[Move.from_dex(move)])
    return BattleState(BattleSide([mon(first)]), BattleSide([mon(second)]))


def step(state, seed=0, **kwargs):
    actions = [MoveAction(s.active_pokemon.moves[0].id, 1) for s in (state.p1,state.p2)]
    return simulate_turn_transition(state, *actions, sample_outcomes=True,
                                    rng=random.Random(seed), **kwargs)


def test_clone_isolates_move_pp_and_nested_boosts():
    parent = battle("swordsdance")
    parent.p1.active_pokemon.last_protect_move = "protect"
    child, sibling = parent.clone(), parent.clone()
    child.p1.active_pokemon.moves[0].pp = 0
    child.p1.active_pokemon.moves[0].self_boosts["atk"] = 6
    child.p1.active_pokemon.raw_stats["atk"] = 1
    assert parent.p1.active_pokemon.moves[0].pp == 32
    assert sibling.p1.active_pokemon.moves[0].self_boosts["atk"] == 2
    assert sibling.p1.active_pokemon.raw_stats["atk"] == 256
    assert child.p1.active_pokemon.last_protect_move == "protect"


def test_miss_consumes_pp_without_mutating_parent_or_applying_effects():
    state = battle("closecombat")
    state.p1.active_pokemon.moves[0].accuracy = 0
    after = step(state)
    assert after.p2.active_pokemon.current_hp == 461
    assert after.p1.active_pokemon.moves[0].pp == state.p1.active_pokemon.moves[0].pp - 1
    assert after.p1.active_pokemon.boosts["def"] == 0


def test_asleep_or_ko_before_acting_does_not_consume_pp():
    state = battle()
    state.p1.active_pokemon.status = StatusCondition.SLEEP
    state.p1.active_pokemon.status_turns = 2
    assert step(state).p1.active_pokemon.moves[0].pp == state.p1.active_pokemon.moves[0].pp
    state = battle("tackle", "tackle", hp=1)
    after = step(state, tie_winner="p1")
    assert after.p2.active_pokemon.moves[0].pp == state.p2.active_pokemon.moves[0].pp


def test_sampled_ties_allow_both_sides_to_win_and_replay_seed():
    state = battle("tackle", "tackle", hp=1)
    winners = [step(state, i).winner for i in range(128)]
    assert 35 < Counter(winners)[1] < 93
    assert winners == [step(state, i).winner for i in range(128)]
    assert state.p1.active_pokemon.current_hp == 1


def test_accuracy_and_damage_have_non_degenerate_distributions():
    state = battle("stoneedge")
    damage = [461-step(state, i).p2.active_pokemon.current_hp for i in range(512)]
    assert 65 < damage.count(0) < 145
    assert len(set(damage)) > 10


def test_protect_is_binary_and_failure_resets_streak():
    first = step(battle("tackle", "protect"))
    assert first.p2.active_pokemon.current_hp == 461
    assert first.p2.active_pokemon.protect_streak == 1
    results = [step(first, i) for i in range(256)]
    blocked = sum(s.p2.active_pokemon.current_hp == 461 for s in results)
    assert 50 < blocked < 120
    for s in results:
        damage = 461-s.p2.active_pokemon.current_hp
        assert damage == 0 or damage >= 67  # never fractional expected damage
        if damage:
            assert s.p2.active_pokemon.protect_streak == 0


def test_protect_blocks_status_and_expires_next_turn():
    state = battle("toxic", "protect")
    after = step(state)
    assert after.p2.active_pokemon.status == StatusCondition.NONE
    after.p2.active_pokemon.moves = [Move.from_dex("amnesia")]
    after.p1.active_pokemon.moves = [Move.from_dex("tackle")]
    assert step(after).p2.active_pokemon.current_hp < after.p2.active_pokemon.current_hp


def test_pressure_cost_and_no_pp_underflow():
    state = battle()
    state.p2.active_pokemon.ability = "Pressure"
    state.p1.active_pokemon.moves[0].pp = 1
    after = step(state)
    assert after.p1.active_pokemon.moves[0].pp == 0
    assert [a.move_id for a in after.get_valid_actions(1)] == ["struggle"]


def test_struggle_can_hit_ghost_and_has_recoil_without_restoring_pp():
    state = battle()
    state.p1.active_pokemon.moves[0].pp = 0
    state.p2.active_pokemon.types = (PokemonType.GHOST, None)
    after = simulate_turn_transition(state, state.get_valid_actions(1)[0], None,
                                     sample_outcomes=True, rng=random.Random(3))
    assert after.p2.active_pokemon.current_hp < 461
    assert after.p1.active_pokemon.current_hp == 461-115
    assert after.p1.active_pokemon.moves[0].pp == 0


def test_sampled_ko_requires_replacement_without_ghost_switch():
    state = battle("tackle", "tackle", hp=1)
    state.p2.pokemon.append(battle().p2.active_pokemon)
    after = step(state, tie_winner="p1")
    assert after.p2.active_index == 0
    assert after.p2.active_pokemon.is_fainted
    assert after.get_valid_actions(2)[0].target_slot == 2


def test_random_roll_precedes_stab_and_crit_ignores_bad_boosts():
    state = battle()
    a,d = state.p1.active_pokemon,state.p2.active_pokemon
    rolls = calculate_damage_rolls(a,d,a.moves[0])
    # Official Gen 9 Snorlax fixture, atk=256, def=166: floor(53*r/100)*1.5.
    assert rolls == [67,67,69,69,70,70,72,72,73,73,75,75,76,76,78,79]
    critical = calculate_damage_rolls(a,d,a.moves[0],is_critical=True)
    a.boosts["atk"], d.boosts["def"] = -6, 6
    assert calculate_damage_rolls(a,d,a.moves[0],is_critical=True) == critical


def test_search_mode_remains_deterministic_and_branches_are_independent():
    state = battle("tackle", "protect")
    a1,a2 = MoveAction("tackle",1),MoveAction("protect",1)
    first = simulate_turn_transition(state,a1,a2)
    left,right = simulate_turn_transition(first,a1,a2),simulate_turn_transition(first,a1,a2)
    assert left == right
    assert first.p1.active_pokemon.moves[0].pp == state.p1.active_pokemon.moves[0].pp-1


def test_factory_preserves_self_target_and_critical_metadata():
    self_move = Move.create("Swords Dance", PokemonType.NORMAL, MoveCategory.STATUS)
    critical_move = Move.create("Stone Edge", PokemonType.ROCK, MoveCategory.PHYSICAL, base_power=100)
    assert self_move.target == "self" and self_move.always_hits
    assert critical_move.crit_ratio == 2


def test_protect_fails_when_acting_last():
    state = battle("tackle", "protect")
    state.p1.active_pokemon.moves[0].priority = 5
    after = step(state)
    assert after.p2.active_pokemon.protect_streak == 0
    assert after.p2.active_pokemon.protect_success_rate == 0.0
