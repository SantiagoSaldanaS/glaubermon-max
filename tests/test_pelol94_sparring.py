"""Automated sparring and regression audit for Pelol94 match pathologies."""

import pytest
import numpy as np
from glaubermon.core.types import PokemonType, MoveCategory, ActionType, StatusCondition, Hazard
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.actions import MoveAction, SwitchAction
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.data.meta_teams import get_meta_team_balance, build_meta_pokemon, get_meta_pokemon_by_species
from glaubermon.search.subgame_resolver import SubgameResolver
from glaubermon.search.evaluators import HeuristicEvaluator


def test_gliscor_ground_immunity_strictly_dominated():
    """Great Tusk must NEVER click Headlong Rush into Gliscor's Flying type; Ice Spinner is 4x super effective."""
    team1 = get_meta_team_balance()
    gliscor = get_meta_pokemon_by_species("Gliscor")
    assert gliscor is not None
    assert PokemonType.FLYING in gliscor.types

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=0),  # Great Tusk active
        p2=BattleSide(pokemon=[gliscor], active_index=0)
    )

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    hr_prob = sum(p for a, p in zip(actions, strat) if getattr(a, "move_id", "") == "headlongrush")
    ice_prob = sum(p for a, p in zip(actions, strat) if getattr(a, "move_id", "") == "icespinner")

    assert hr_prob < 0.001, f"Headlong Rush into Flying immunity must be 0%, got {hr_prob * 100:.2f}%"
    assert ice_prob > 0.90, f"Ice Spinner into 4x weak Gliscor must be > 90%, got {ice_prob * 100:.2f}%"
    assert action.move_id == "icespinner"


def test_ting_lu_does_not_earthquake_gliscor():
    """Ting-Lu must NEVER click Earthquake into Gliscor's Flying type."""
    team1 = get_meta_team_balance()
    gliscor = get_meta_pokemon_by_species("Gliscor")

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=5),  # Ting-Lu active
        p2=BattleSide(pokemon=[gliscor], active_index=0)
    )

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    eq_prob = next((p for a, p in zip(actions, strat) if getattr(a, "move_id", "") == "earthquake"), 0.0)
    assert eq_prob < 0.001, f"Earthquake into Flying immunity must be 0%, got {eq_prob * 100:.2f}%"
    assert getattr(action, "move_id", "") != "earthquake"


def test_kingambit_early_switch_rejection():
    """Turn 7 Regression: With only 1 ally fainted, Ogerpon must NOT switch in Kingambit to take chip."""
    team1 = get_meta_team_balance()
    team1[3].current_hp = 0  # Only Dragapult fainted (1 fallen)
    team1[4].current_hp = int(team1[4].max_hp * 0.20)  # Ogerpon at 20%

    opp_drag = build_meta_pokemon("Dragapult", ["Shadow Ball", "Draco Meteor", "Flamethrower", "U-turn"], "Choice Specs", "Ghost", "fast_spec")
    opp_drag.current_hp = int(opp_drag.max_hp * 0.41)
    opp_drag.is_terastallized = True
    opp_drag.tera_type = PokemonType.GHOST

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=4),  # Ogerpon active
        p2=BattleSide(pokemon=[opp_drag], active_index=0)
    )

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    gambit_prob = next((p for a, p in zip(actions, strat) if getattr(a, "species", "") == "Kingambit"), 0.0)
    assert gambit_prob < 0.01, f"Early Kingambit switch probability must be < 1%, got {gambit_prob * 100:.2f}%"


def test_kingambit_pivots_to_air_balloon_gholdengo_vs_gliscor():
    """Turn 9 Regression: Kingambit facing Gliscor Earthquake must pivot to immune Air Balloon Gholdengo."""
    team1 = get_meta_team_balance()
    team1[2].current_hp = 170  # Kingambit in lethal range of Earthquake (178-210)
    # Gholdengo has Air Balloon (immune to Ground)
    assert "airballoon" in team1[1].item.lower().replace(" ", "")

    gliscor = get_meta_pokemon_by_species("Gliscor")
    gliscor.current_hp = int(gliscor.max_hp * 0.88)

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=2, is_tera_used=True),  # Kingambit active, Tera already used
        p2=BattleSide(pokemon=[gliscor], active_index=0)
    )

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    # Kingambit must NOT stay in to take 51% Earthquake; should pivot to Air Balloon Gholdengo
    ghold_prob = next((p for a, p in zip(actions, strat) if getattr(a, "species", "") == "Gholdengo"), 0.0)
    assert ghold_prob > 0.50, f"Pivot to Air Balloon Gholdengo must be > 50%, got {ghold_prob * 100:.2f}%"


def test_tera_option_preservation_on_crippled_target():
    """Turn 4 Regression: Dragapult facing 12% HP Iron Valiant must NOT burn team Terastallization."""
    team1 = get_meta_team_balance()
    opp_valiant = get_meta_pokemon_by_species("Iron Valiant")
    opp_valiant.current_hp = int(opp_valiant.max_hp * 0.12)  # 12% HP left

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=3),  # Dragapult active
        p2=BattleSide(pokemon=[opp_valiant], active_index=0)
    )

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    tera_prob = sum(p for a, p in zip(actions, strat) if getattr(a, "is_tera", False))
    assert tera_prob < 0.01, f"Terastallizing on a 12% HP target must be < 1%, got {tera_prob * 100:.2f}%"


def test_opponent_tera_isolation_state_sync():
    """Verify that when opponent Dragapult terastallizes, subsequent switch-in Gliscor retains Ground/Flying."""
    from glaubermon.client.showdown_bot import ShowdownBot

    bot = ShowdownBot(username="TestBot", checkpoint="checkpoints/glaubermon_rebel_latest.pt")
    room = "battle-test-room"

    # Step 1: User identifies sides
    bot.our_side[room] = "p1"

    # Step 2: Opponent Dragapult terastallizes to Ghost
    bot.ident_to_species[room] = {"p2a: Dragapult": "Dragapult"}
    bot.opp_tera[room] = "Ghost"
    bot.opp_tera_mon[room] = "Dragapult"

    # Step 3: Opponent switches to Gliscor
    bot.ident_to_species[room]["p2a: Gliscor"] = "Gliscor"
    bot.opp_active[room] = "Gliscor"
    bot.opp_team[room] = ["Dragapult", "Gliscor"]

    # Build simulated request for bot
    req = {
        "side": {
            "pokemon": [
                {"details": "Great Tusk, L100", "condition": "361/361", "active": True, "moves": ["closecombat", "headlongrush", "icespinner", "rapidspin"], "item": "boosterenergy"}
            ]
        },
        "active": [{"moves": [{"id": "closecombat"}, {"id": "headlongrush"}, {"id": "icespinner"}, {"id": "rapidspin"}]}]
    }

    state = bot.build_battle_state(room, req)
    p2_active = state.p2.active_pokemon

    assert p2_active.species == "Gliscor"
    assert not p2_active.is_terastallized, f"Gliscor must NOT be marked as terastallized, got {p2_active.is_terastallized}"
    assert p2_active.active_types == (PokemonType.GROUND, PokemonType.FLYING), f"Gliscor must be Ground/Flying, got {p2_active.active_types}"
