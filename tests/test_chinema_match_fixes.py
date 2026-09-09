import pytest
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, StatusCondition, ActionType
from glaubermon.data.meta_teams import get_meta_pokemon_by_species
from glaubermon.search.subgame_resolver import SubgameResolver
from glaubermon.search.evaluators import HeuristicEvaluator
from glaubermon.client.showdown_bot import ShowdownBot


def test_opponent_air_balloon_popped_enables_earthquake():
    """Turn 17-21 Autopsy: Ting-Lu must choose Earthquake (guaranteed OHKO) over Ruination
    when opponent Gholdengo has had its Air Balloon popped."""
    tinglu = get_meta_pokemon_by_species("Ting-Lu")
    
    # Gholdengo at 50% HP with popped Air Balloon (item=None)
    gholdengo_popped = get_meta_pokemon_by_species("Gholdengo")
    gholdengo_popped.item = None
    gholdengo_popped.current_hp = int(gholdengo_popped.max_hp * 0.50)

    state = BattleState(
        p1=BattleSide(pokemon=[tinglu], active_index=0),
        p2=BattleSide(pokemon=[gholdengo_popped], active_index=0)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, p1_dist, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    assert action.action_type == ActionType.MOVE
    assert action.move_id == "earthquake", f"Expected Earthquake OHKO, but got {action.move_id}"
    eq_prob = next((p for a, p in zip(actions, p1_dist) if getattr(a, "move_id", "") == "earthquake"), 0.0)
    assert eq_prob > 0.95, f"Earthquake probability should be near 1.0, got {eq_prob}"


def test_tusk_ice_spinner_vs_tera_flying_kingambit():
    """Turn 28 Autopsy: When Kingambit can Terastallize to Flying, Great Tusk must actively
    mix/favor Ice Spinner and not blindly click resisted Close Combat."""
    tusk = get_meta_pokemon_by_species("Great Tusk")
    tusk.boosts["atk"] = 1

    kingambit = get_meta_pokemon_by_species("Kingambit")
    kingambit.tera_type = PokemonType.FLYING
    kingambit.current_hp = int(kingambit.max_hp * 0.65)

    state = BattleState(
        p1=BattleSide(pokemon=[tusk], active_index=0),
        p2=BattleSide(pokemon=[kingambit], active_index=0)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, p1_dist, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    ice_prob = sum(p for a, p in zip(actions, p1_dist) if getattr(a, "move_id", "") == "icespinner")
    assert ice_prob > 0.25, f"Ice Spinner should be heavily represented in Nash equilibrium vs Tera Flying, got {ice_prob}"

    # Also test when Kingambit is ALREADY Terastallized to Flying: Ice Spinner must be 100%
    kingambit_tera = kingambit.clone()
    kingambit_tera.is_terastallized = True
    state_tera = BattleState(
        p1=BattleSide(pokemon=[tusk], active_index=0),
        p2=BattleSide(pokemon=[kingambit_tera], active_index=0, is_tera_used=True)
    )
    action_tera, p1_dist_tera, actions_tera, val_tera = resolver.resolve_turn(state_tera, depth=1, sample=False)
    assert action_tera.action_type == ActionType.MOVE
    assert action_tera.move_id == "icespinner", f"Expected Ice Spinner vs active Flying type, got {action_tera.move_id}"


def test_close_combat_over_ice_spinner_on_ting_lu():
    """Turn 13-15 Autopsy: Great Tusk against Ting-Lu with popped Gholdengo on bench
    must use Close Combat / Headlong Rush, completely rejecting low-damage Ice Spinner."""
    tusk = get_meta_pokemon_by_species("Great Tusk")
    tusk.boosts["atk"] = 1

    tinglu = get_meta_pokemon_by_species("Ting-Lu")
    gholdengo_popped = get_meta_pokemon_by_species("Gholdengo")
    gholdengo_popped.item = None
    gholdengo_popped.current_hp = int(gholdengo_popped.max_hp * 0.81)
    kingambit = get_meta_pokemon_by_species("Kingambit")

    state = BattleState(
        p1=BattleSide(pokemon=[tusk], active_index=0),
        p2=BattleSide(pokemon=[tinglu, gholdengo_popped, kingambit], active_index=0)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, p1_dist, actions, val = resolver.resolve_turn(state, depth=2, sample=False)

    ice_prob = sum(p for a, p in zip(actions, p1_dist) if getattr(a, "move_id", "") == "icespinner")
    assert ice_prob < 0.05, f"Ice Spinner must be <= 5% vs Ting-Lu when STABs are viable, got {ice_prob}"
    cc_or_hlr = sum(p for a, p in zip(actions, p1_dist) if getattr(a, "move_id", "") in ("closecombat", "headlongrush"))
    assert cc_or_hlr > 0.90, f"Close Combat / Headlong Rush should dominate (>= 90%), got {cc_or_hlr}"


def test_protect_mirror_switch_preference():
    """Turn 5, 7, 9 Autopsy: Ogerpon vs Ogerpon mirror match with healthy Dragapult on bench
    must prioritize switching to Dragapult rather than looping failing Spiky Shield."""
    ogerpon_p1 = get_meta_pokemon_by_species("Ogerpon-Wellspring")
    ogerpon_p1.current_hp = int(ogerpon_p1.max_hp * 0.45)
    dragapult_p1 = get_meta_pokemon_by_species("Dragapult")

    ogerpon_p2 = get_meta_pokemon_by_species("Ogerpon-Wellspring")
    ogerpon_p2.current_hp = int(ogerpon_p2.max_hp * 0.45)

    state = BattleState(
        p1=BattleSide(pokemon=[ogerpon_p1, dragapult_p1], active_index=0),
        p2=BattleSide(pokemon=[ogerpon_p2], active_index=0)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, p1_dist, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    protect_prob = sum(p for a, p in zip(actions, p1_dist) if getattr(a, "move_id", "") == "spikyshield")
    assert protect_prob < 0.05, f"Spiky Shield should NOT be spammed in mirror match with hard counter on bench, got {protect_prob}"
    sw_prob = sum(p for a, p in zip(actions, p1_dist) if a.action_type == ActionType.SWITCH)
    assert sw_prob > 0.60, f"Switching to Dragapult counter should be favored (> 60%), got {sw_prob}"


def test_showdown_bot_protect_streak_isolation():
    """Verify that ShowdownBot tracks protect streaks per side, preventing mirror contamination."""
    import asyncio
    bot = ShowdownBot(username="Glaubermax")
    room = "battle-gen9ou-1"
    bot.our_side[room] = "p1"
    bot.ident_to_species[room] = {
        "p1a: Ogerpon": "Ogerpon-Wellspring",
        "p2a: Ogerpon": "Ogerpon-Wellspring"
    }

    # P1 (us) uses Spiky Shield
    asyncio.run(bot.handle_message(f">{room}\n|move|p1a: Ogerpon|Spiky Shield|p1a: Ogerpon"))
    assert bot.protect_streaks[room]["p1"]["ogerpon"] == 1
    assert bot.protect_streaks[room].get("p2", {}).get("ogerpon", 0) == 0

    # P2 (opponent) uses Horn Leech
    asyncio.run(bot.handle_message(f">{room}\n|move|p2a: Ogerpon|Horn Leech|p1a: Ogerpon"))
    assert bot.protect_streaks[room]["p1"]["ogerpon"] == 1
    assert bot.protect_streaks[room]["p2"]["ogerpon"] == 0

    # P1 switches out to Gholdengo
    asyncio.run(bot.handle_message(f">{room}\n|switch|p1a: Gholdengo|Gholdengo, L100|100/100"))
    assert bot.protect_streaks[room]["p1"]["ogerpon"] == 0
    assert bot.protect_streaks[room]["p1"]["gholdengo"] == 0
