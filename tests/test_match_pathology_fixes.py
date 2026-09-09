import pytest
import numpy as np
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, StatusCondition, ActionType
from glaubermon.core.battle_state import BattleState, BattleSide, MoveAction, SwitchAction
from glaubermon.search.subgame_resolver import SubgameResolver
from glaubermon.search.evaluators import HeuristicEvaluator


def build_test_mon(name, types, hp=300, moves=None, item=None, ability=None, stats=None, status=StatusCondition.NONE):
    if moves is None:
        moves = [
            Move.create("Tackle", PokemonType.NORMAL, MoveCategory.PHYSICAL, 40, 100),
        ]
    st = {"hp": hp, "atk": 250, "def": 250, "spa": 250, "spd": 250, "spe": 200}
    if stats:
        st.update(stats)
    return Pokemon(
        species=name,
        types=types,
        raw_stats=st,
        current_hp=hp,
        max_hp=hp,
        status=status,
        moves=moves,
        item=item,
        ability=ability
    )


def test_whirlwind_preferred_over_ruination_against_gliscor_spikes():
    """Ting-Lu must prefer Whirlwind over Ruination when facing Gliscor laying Spikes."""
    tinglu_moves = [
        Move.create("Earthquake", PokemonType.GROUND, MoveCategory.PHYSICAL, 100, 100),
        Move.create("Ruination", PokemonType.DARK, MoveCategory.SPECIAL, 1, 90),
        Move.create("Stealth Rock", PokemonType.ROCK, MoveCategory.STATUS, 0, 100),
        Move.create("Whirlwind", PokemonType.NORMAL, MoveCategory.STATUS, 0, 100),
    ]
    tinglu = build_test_mon("Ting-Lu", (PokemonType.DARK, PokemonType.GROUND), hp=450, moves=tinglu_moves, item="Leftovers", stats={"spe": 120})

    gliscor_moves = [
        Move.create("Spikes", PokemonType.GROUND, MoveCategory.STATUS, 0, 100),
        Move.create("Protect", PokemonType.NORMAL, MoveCategory.STATUS, 0, 100),
        Move.create("Toxic", PokemonType.POISON, MoveCategory.STATUS, 0, 90),
        Move.create("Earthquake", PokemonType.GROUND, MoveCategory.PHYSICAL, 100, 100),
    ]
    gliscor = build_test_mon("Gliscor", (PokemonType.GROUND, PokemonType.FLYING), hp=280, moves=gliscor_moves, item="Toxic Orb", ability="Poison Heal", status=StatusCondition.TOXIC, stats={"spe": 210})
    gliscor.current_hp = 200  # ~71% HP

    bench_p1 = build_test_mon("Great Tusk", (PokemonType.GROUND, PokemonType.FIGHTING), hp=350)
    bench_p2 = build_test_mon("Dragapult", (PokemonType.DRAGON, PokemonType.GHOST), hp=300)

    state = BattleState(
        p1=BattleSide(pokemon=[tinglu, bench_p1], active_index=0),
        p2=BattleSide(pokemon=[gliscor, bench_p2], active_index=0),
        turn=11
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    # Subgame with Gliscor clicking Spikes
    action, strat, actions, val = resolver.resolve_turn(
        state, depth=1, sample=False,
        p1_actions_override=[
            MoveAction(move_id="ruination", move_slot=2),
            MoveAction(move_id="whirlwind", move_slot=4)
        ]
    )
    # Whirlwind must be chosen over Ruination against hazard-setting Gliscor
    assert action.move_id == "whirlwind"


def test_redundant_tera_penalized_when_base_move_kos():
    """Great Tusk must not waste Tera Ice when unboosted Headlong Rush already secures guaranteed KO."""
    tusk_moves = [
        Move.create("Headlong Rush", PokemonType.GROUND, MoveCategory.PHYSICAL, 120, 100),
        Move.create("Ice Spinner", PokemonType.ICE, MoveCategory.PHYSICAL, 80, 100),
    ]
    tusk = build_test_mon("Great Tusk", (PokemonType.GROUND, PokemonType.FIGHTING), hp=300, moves=tusk_moves, stats={"atk": 360, "spe": 270})
    tusk.tera_type = PokemonType.ICE

    # Crippled Dragapult at low HP (30 HP) where Headlong Rush easily KOs without Tera
    dragapult_moves = [
        Move.create("Draco Meteor", PokemonType.DRAGON, MoveCategory.SPECIAL, 130, 90),
    ]
    dragapult = build_test_mon("Dragapult", (PokemonType.DRAGON, PokemonType.GHOST), hp=280, moves=dragapult_moves, stats={"def": 180, "spe": 380})
    dragapult.current_hp = 30
    dragapult.boosts["spa"] = -2  # Stat-dropped

    state = BattleState(
        p1=BattleSide(pokemon=[tusk], active_index=0),
        p2=BattleSide(pokemon=[dragapult], active_index=0),
        turn=22
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(
        state, depth=1, sample=False,
        p1_actions_override=[
            MoveAction(move_id="headlongrush", move_slot=1, is_tera=False),
            MoveAction(move_id="headlongrush", move_slot=1, is_tera=True, tera_type=PokemonType.ICE),
        ]
    )
    # Must choose non-Tera version to avoid squandering Tera
    assert action.is_tera is False
    assert action.move_id == "headlongrush"


def test_sucker_switch_carousel_penalized_on_low_hp_bench():
    """Dragapult must attack rather than switching when all bench members are low HP against Kingambit."""
    draga_moves = [
        Move.create("Flamethrower", PokemonType.FIRE, MoveCategory.SPECIAL, 90, 100),
        Move.create("Shadow Ball", PokemonType.GHOST, MoveCategory.SPECIAL, 80, 100),
    ]
    dragapult = build_test_mon("Dragapult", (PokemonType.DRAGON, PokemonType.GHOST), hp=280, moves=draga_moves, stats={"spe": 380})
    dragapult.current_hp = 20  # ~7% HP

    # Bench members are critically low
    bench_tinglu = build_test_mon("Ting-Lu", (PokemonType.DARK, PokemonType.GROUND), hp=450)
    bench_tinglu.current_hp = 35  # ~8% HP

    bench_gambit = build_test_mon("Kingambit", (PokemonType.DARK, PokemonType.STEEL), hp=340)
    bench_gambit.current_hp = 25  # ~7% HP

    opp_gambit_moves = [
        Move.create("Sucker Punch", PokemonType.DARK, MoveCategory.PHYSICAL, 70, 100, priority=1),
        Move.create("Kowtow Cleave", PokemonType.DARK, MoveCategory.PHYSICAL, 85, 100),
    ]
    opp_gambit = build_test_mon("Kingambit", (PokemonType.DARK, PokemonType.STEEL), hp=340, moves=opp_gambit_moves, stats={"atk": 400, "spe": 130})

    state = BattleState(
        p1=BattleSide(pokemon=[dragapult, bench_tinglu, bench_gambit], active_index=0),
        p2=BattleSide(pokemon=[opp_gambit], active_index=0),
        turn=37
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(
        state, depth=1, sample=False,
        p1_actions_override=[
            MoveAction(move_id="flamethrower", move_slot=1),
            SwitchAction(target_slot=2, species="Ting-Lu"),
            SwitchAction(target_slot=3, species="Kingambit")
        ]
    )
    # Dragapult must commit to an attack, NOT switch out to die
    assert action.action_type == ActionType.MOVE
    assert action.move_id == "flamethrower"


def test_sucker_punch_penalized_against_protect_streak():
    """Kingambit must prefer Kowtow Cleave over Sucker Punch when facing an opponent protecting."""
    gambit_moves = [
        Move.create("Sucker Punch", PokemonType.DARK, MoveCategory.PHYSICAL, 70, 100, priority=1),
        Move.create("Kowtow Cleave", PokemonType.DARK, MoveCategory.PHYSICAL, 85, 100),
    ]
    gambit = build_test_mon("Kingambit", (PokemonType.DARK, PokemonType.STEEL), hp=340, moves=gambit_moves, stats={"atk": 380, "spe": 130})
    gambit.current_hp = 40  # 12% HP

    oger_moves = [
        Move.create("Spiky Shield", PokemonType.GRASS, MoveCategory.STATUS, 0, 100, priority=4),
        Move.create("Horn Leech", PokemonType.GRASS, MoveCategory.PHYSICAL, 75, 100),
    ]
    ogerpon = build_test_mon("Ogerpon", (PokemonType.GRASS, PokemonType.WATER), hp=300, moves=oger_moves, stats={"spe": 350})
    ogerpon.protect_streak = 1  # Already used protect

    state = BattleState(
        p1=BattleSide(pokemon=[gambit], active_index=0),
        p2=BattleSide(pokemon=[ogerpon], active_index=0),
        turn=22
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(
        state, depth=1, sample=False,
        p1_actions_override=[
            MoveAction(move_id="suckerpunch", move_slot=1),
            MoveAction(move_id="kowtowcleave", move_slot=2),
        ],
        p1_sucker_streak=1
    )
    # Kowtow Cleave must be chosen over repeatedly failing Sucker Punch
    assert action.move_id == "kowtowcleave"


def test_move_slot_indices_preserved_with_disabled_moves():
    """Disabled move in slot 1 must result in slots 2, 3, 4 having move_slot 2, 3, 4."""
    moves = [
        Move.create("Disabled Move", PokemonType.NORMAL, MoveCategory.STATUS, 0, 100),
        Move.create("Move Two", PokemonType.FIRE, MoveCategory.SPECIAL, 90, 100),
        Move.create("Move Three", PokemonType.WATER, MoveCategory.SPECIAL, 90, 100),
        Move.create("Move Four", PokemonType.GRASS, MoveCategory.PHYSICAL, 90, 100),
    ]
    # Mark slot 0 as disabled with pp = 0
    moves[0].pp = 0

    mon = build_test_mon("TestMon", (PokemonType.NORMAL,), moves=moves)
    state = BattleState(
        p1=BattleSide(pokemon=[mon], active_index=0),
        p2=BattleSide(pokemon=[build_test_mon("OppMon", (PokemonType.NORMAL,))], active_index=0),
        turn=1
    )
    valid_acts = [a for a in state.get_valid_actions(player=1) if a.action_type == ActionType.MOVE]
    slots = [a.move_slot for a in valid_acts]
    assert 1 not in slots  # Slot 1 is disabled and omitted
    assert slots == [2, 3, 4]  # Slots 2, 3, 4 correctly preserved!


def test_anti_carousel_gholdengo_immune_gliscor_does_not_switch():
    """Air Balloon Gholdengo facing Gliscor at Depth 2 must NOT switch to Ting-Lu (breaks ping-pong loop)."""
    from glaubermon.data.meta_teams import get_meta_team_balance, get_meta_team_pelol94
    from glaubermon.core.types import Hazard
    team1 = get_meta_team_balance()
    team2 = get_meta_team_pelol94()
    gliscor_idx = next(i for i, p in enumerate(team2) if p.species == "Gliscor")

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=1),  # Gholdengo active
        p2=BattleSide(pokemon=team2, active_index=gliscor_idx),
        turn=5
    )
    state.p1.hazards[Hazard.SPIKES_1] = 2

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=2, sample=False, last_action_was_switch=True)
    assert action.action_type == ActionType.MOVE, f"Gholdengo must attack or set up, got switch: {action}"
    assert getattr(action, "move_id", "") in ("shadowball", "makeitrain", "nastyplot")


def test_defensive_tera_anticipation_great_tusk_selects_headlong_rush():
    """Great Tusk facing Ting-Lu (with Tera Poison unused) must pick Headlong Rush over Close Combat."""
    from glaubermon.data.meta_teams import get_meta_team_balance, get_meta_team_pelol94
    team1 = get_meta_team_balance()
    team2 = get_meta_team_pelol94()
    tinglu_idx = next(i for i, p in enumerate(team2) if p.species == "Ting-Lu")

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=0, is_tera_used=True),  # Great Tusk active
        p2=BattleSide(pokemon=team2, active_index=tinglu_idx, is_tera_used=False),
        turn=15
    )

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=2, sample=False)
    assert getattr(action, "move_id", "") == "headlongrush", f"Expected Headlong Rush into potential Tera Poison, got: {action}"


def test_kingambit_healthy_foe_selects_high_power_stab_over_sucker_punch():
    """Kingambit facing 100% HP opponent must select high BP STAB attack over non-lethal Sucker Punch."""
    from glaubermon.data.meta_teams import get_meta_team_balance, get_meta_team_pelol94
    team1 = get_meta_team_balance()
    team2 = get_meta_team_pelol94()
    k_idx1 = next(i for i, p in enumerate(team1) if p.species == "Kingambit")
    k_idx2 = next(i for i, p in enumerate(team2) if p.species == "Kingambit")

    # Opponent Kingambit at 100% HP
    team2[k_idx2].current_hp = team2[k_idx2].max_hp
    team1[k_idx1].current_hp = int(team1[k_idx1].max_hp * 0.35)

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=k_idx1, is_tera_used=True),
        p2=BattleSide(pokemon=team2, active_index=k_idx2, is_tera_used=True),
        turn=26
    )

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=2, sample=False)
    assert getattr(action, "move_id", "") in ("kowtowcleave", "ironhead"), f"Expected high power STAB, got: {action}"
    assert getattr(action, "move_id", "") != "suckerpunch"


def test_ogerpon_rejects_spiky_shield_into_setup_sweeper():
    """Ogerpon facing Kingambit (with Swords Dance) must attack directly instead of clicking Spiky Shield."""
    from glaubermon.data.meta_teams import get_meta_team_balance, get_meta_team_pelol94
    team1 = get_meta_team_balance()
    team2 = get_meta_team_pelol94()
    og_idx = next(i for i, p in enumerate(team1) if "Ogerpon" in p.species)
    k_idx2 = next(i for i, p in enumerate(team2) if p.species == "Kingambit")

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=og_idx, is_tera_used=True),
        p2=BattleSide(pokemon=team2, active_index=k_idx2, is_tera_used=True),
        turn=27
    )

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=2, sample=False)
    assert getattr(action, "move_id", "") != "spikyshield", f"Spiky Shield gives opponent free Swords Dance: {action}"
    assert getattr(action, "move_id", "") in ("ivycudgel", "hornleech", "playrough")

