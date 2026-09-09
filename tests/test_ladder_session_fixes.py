"""Unit tests verifying engine improvements from post-ladder match analysis."""

import pytest
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, ActionType, Weather, Terrain
from glaubermon.core.actions import MoveAction, SwitchAction
from glaubermon.inference.damage_calc import calculate_damage_rolls
from glaubermon.data.showdown_dex import ShowdownDex
from glaubermon.search.subgame_resolver import SubgameResolver
from glaubermon.search.evaluators import HeuristicEvaluator
from glaubermon.client.showdown_bot import ShowdownBot


def test_hydro_steam_in_sun():
    """Verify Gen 9 Hydro Steam gets 1.5x damage in Sun rather than 0.5x halving."""
    dex = ShowdownDex.get_instance()
    ww = Pokemon(species="Walking Wake", current_hp=339, max_hp=339, types=(PokemonType.WATER, PokemonType.DRAGON))
    target = Pokemon(species="Kingambit", current_hp=341, max_hp=341, types=(PokemonType.DARK, PokemonType.STEEL))
    hydro_steam = dex.get_move("hydrosteam")

    r_none = calculate_damage_rolls(ww, target, hydro_steam, weather=Weather.NONE)
    r_sun = calculate_damage_rolls(ww, target, hydro_steam, weather=Weather.SUN)
    r_rain = calculate_damage_rolls(ww, target, hydro_steam, weather=Weather.RAIN)

    # In Sun, Hydro Steam must do significantly MORE than without Sun (1.5x vs 1.0x)
    assert min(r_sun) > min(r_none)
    # In Rain, Hydro Steam also gets 1.5x boost
    assert min(r_rain) > min(r_none)
    # Ratio between Sun and None should be ~1.5x
    assert min(r_sun) >= int(min(r_none) * 1.45)


def test_weather_and_terrain_tracking():
    """Verify ShowdownBot parses weather and terrain protocol messages into BattleState."""
    bot = ShowdownBot(username="Glaubermax", depth=1)
    room = "battle-gen9ou-test123"

    # Simulate Showdown protocol messages
    import asyncio
    asyncio.run(bot.handle_message(f">{room}\n|-weather|SunnyDay|[from] ability: Drought|[of] p2a: Ninetales"))
    asyncio.run(bot.handle_message(f">{room}\n|-fieldstart|move: Grassy Terrain"))

    assert bot.room_weather.get(room) == Weather.SUN
    assert bot.room_terrain.get(room) == Terrain.GRASSY

    # Verify weather ends
    asyncio.run(bot.handle_message(f">{room}\n|-weather|none"))
    assert bot.room_weather.get(room) == Weather.NONE

    # Set rain and electric terrain
    asyncio.run(bot.handle_message(f">{room}\n|-weather|RainDance"))
    asyncio.run(bot.handle_message(f">{room}\n|-fieldstart|move: Electric Terrain"))
    assert bot.room_weather.get(room) == Weather.RAIN
    assert bot.room_terrain.get(room) == Terrain.ELECTRIC

    # Check terrain end
    asyncio.run(bot.handle_message(f">{room}\n|-fieldend|move: Electric Terrain"))
    assert bot.room_terrain.get(room) == Terrain.NONE


def test_recovery_overheal_blunder_penalty():
    """Verify Gholdengo at healthy HP (81%) will not select Recover against an offensive threat."""
    dex = ShowdownDex.get_instance()
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())

    # Gholdengo at 255/315 HP (81%)
    gholdengo = Pokemon(
        species="Gholdengo",
        current_hp=255,
        max_hp=315,
        types=(PokemonType.STEEL, PokemonType.GHOST),
        moves=[
            dex.get_move("makeitrain"),
            dex.get_move("shadowball"),
            dex.get_move("nastyplot"),
            dex.get_move("recover"),
        ]
    )
    # Ting-Lu and Kingambit on bench
    tinglu = Pokemon(species="Ting-Lu", current_hp=400, max_hp=514, types=(PokemonType.DARK, PokemonType.GROUND))
    kingambit = Pokemon(species="Kingambit", current_hp=341, max_hp=341, types=(PokemonType.DARK, PokemonType.STEEL))

    # Walking Wake active in Sun
    ww = Pokemon(
        species="Walking Wake",
        current_hp=339,
        max_hp=339,
        types=(PokemonType.WATER, PokemonType.DRAGON),
        moves=[dex.get_move("hydrosteam"), dex.get_move("flamethrower")]
    )

    state = BattleState(
        p1=BattleSide(pokemon=[gholdengo, tinglu, kingambit], active_index=0),
        p2=BattleSide(pokemon=[ww], active_index=0),
        weather=Weather.SUN,
        turn=12
    )

    action, strategy, actions, val = resolver.resolve_turn(state, depth=1, sample=False)
    # Recover should NOT be the chosen action when facing a lethal special attacker at 81% HP
    assert getattr(action, "move_id", "") != "recover"


def test_early_game_tera_preservation():
    """Verify Ogerpon-Wellspring preserves Tera on Turn 1 against a full HP opponent."""
    dex = ShowdownDex.get_instance()
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())

    ogerpon = Pokemon(
        species="Ogerpon-Wellspring",
        current_hp=301,
        max_hp=301,
        types=(PokemonType.GRASS, PokemonType.WATER),
        tera_type=PokemonType.WATER,
        moves=[
            dex.get_move("ivycudgel"),
            dex.get_move("hornleech"),
            dex.get_move("playrough"),
            dex.get_move("spikyshield")
        ]
    )
    ninetales = Pokemon(
        species="Ninetales",
        current_hp=287,
        max_hp=287,
        types=(PokemonType.FIRE, None),
        moves=[dex.get_move("weatherball"), dex.get_move("solarbeam")]
    )

    tusk = Pokemon(species="Great Tusk", current_hp=371, max_hp=371, types=(PokemonType.GROUND, PokemonType.FIGHTING))
    ghold = Pokemon(species="Gholdengo", current_hp=315, max_hp=315, types=(PokemonType.STEEL, PokemonType.GHOST))
    king = Pokemon(species="Kingambit", current_hp=341, max_hp=341, types=(PokemonType.DARK, PokemonType.STEEL))
    drag = Pokemon(species="Dragapult", current_hp=317, max_hp=317, types=(PokemonType.DRAGON, PokemonType.GHOST))
    ting = Pokemon(species="Ting-Lu", current_hp=514, max_hp=514, types=(PokemonType.DARK, PokemonType.GROUND))

    state = BattleState(
        p1=BattleSide(pokemon=[ogerpon, tusk, ghold, king, drag, ting], active_index=0, is_tera_used=False),
        p2=BattleSide(pokemon=[ninetales], active_index=0),
        weather=Weather.SUN,
        turn=1
    )

    action, strategy, actions, val = resolver.resolve_turn(state, depth=1, sample=False)
    # On Turn 1 into Sun against full HP target, burning Tera should be preserved
    assert getattr(action, "is_tera", False) is False


def test_faster_lethal_priority_preference():
    """Verify Kingambit chooses Sucker Punch over slower Kowtow Cleave when facing a faster lethal threat."""
    dex = ShowdownDex.get_instance()
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())

    # Kingambit at low HP facing a faster Walking Wake with lethal move
    kingambit = Pokemon(
        species="Kingambit",
        current_hp=120,
        max_hp=341,
        types=(PokemonType.DARK, PokemonType.STEEL),
        moves=[
            dex.get_move("kowtowcleave"),
            dex.get_move("suckerpunch"),
            dex.get_move("ironhead"),
            dex.get_move("swordsdance")
        ]
    )
    ww = Pokemon(
        species="Walking Wake",
        current_hp=200,
        max_hp=339,
        types=(PokemonType.WATER, PokemonType.DRAGON),
        moves=[dex.get_move("hydrosteam"), dex.get_move("flamethrower")]
    )

    state = BattleState(
        p1=BattleSide(pokemon=[kingambit], active_index=0),
        p2=BattleSide(pokemon=[ww], active_index=0),
        weather=Weather.SUN,
        turn=14
    )

    action, strategy, actions, val = resolver.resolve_turn(state, depth=1, sample=False)
    # Walking Wake outspeeds and lethal Hydro Steam/Flamethrower KOs Kingambit.
    # Kingambit must prioritize Sucker Punch so it strikes before fainting!
    assert getattr(action, "move_id", "") == "suckerpunch"
