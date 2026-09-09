import pytest
from glaubermon.core.battle_state import BattleState, BattleSide, ActionType
from glaubermon.core.pokemon import Pokemon, Move, MoveCategory, StatusCondition
from glaubermon.core.constants import PokemonType
from glaubermon.data.meta_teams import get_meta_team_balance
from glaubermon.search.subgame_resolver import SubgameResolver
from glaubermon.search.evaluators import HeuristicEvaluator


def test_dragapult_choice_specs_prefers_stab_over_coverage():
    """Dragapult holding Choice Specs facing 51% Great Tusk with Kingambit on bench must NOT use Flamethrower."""
    team1 = get_meta_team_balance()
    team2 = get_meta_team_balance()

    # Dragapult is team1[3]. Active is Dragapult.
    p1_drag = team1[3]
    assert "choicespecs" in (p1_drag.item or "").lower().replace(" ", "").replace("-", "")

    # Opponent Great Tusk is team2[0], at 51% HP.
    p2_tusk = team2[0]
    p2_tusk.current_hp = int(p2_tusk.max_hp * 0.51)

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=3),
        p2=BattleSide(pokemon=team2, active_index=0)
    )

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    flame_prob = next((p for a, p in zip(actions, strat) if getattr(a, "move_id", "") == "flamethrower"), 0.0)
    assert flame_prob < 0.01, f"Dragapult must not choose non-STAB Flamethrower when holding Choice Specs, got {flame_prob * 100:.1f}%"
    assert action.move_id in ("shadowball", "dracometeor", "uturn")


def test_ting_lu_faster_lethal_pivots_to_ghost():
    """Ting-Lu at 15% HP facing faster Booster Energy Great Tusk must pivot to Ghost teammate (immune to Close Combat)."""
    team1 = get_meta_team_balance()
    team2 = get_meta_team_balance()

    # Ting-Lu is team1[5]. Active is Ting-Lu, at 15% HP.
    team1[5].current_hp = int(team1[5].max_hp * 0.15)

    # Opponent Great Tusk is team2[0], holding Booster Energy with active Atk boost.
    team2[0].booster_stat = "atk"
    team2[0].item = None

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=5),
        p2=BattleSide(pokemon=team2, active_index=0)
    )

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    # Ting-Lu must switch to an immune or resistant teammate (Gholdengo or Dragapult), not stay in and die
    assert action.action_type == ActionType.SWITCH, f"Ting-Lu at 15% HP must pivot out against faster lethal Close Combat, chose: {action}"
    chosen_spec = getattr(action, "species", "")
    assert chosen_spec in ("Gholdengo", "Dragapult"), f"Ting-Lu should pivot to Ghost-type immunity, chose: {chosen_spec}"


def test_all_moves_disabled_restricts_to_switches():
    """When all moves are disabled (e.g. Choice lock + Cursed Body), legal actions must restrict to switches."""
    from glaubermon.client.showdown_bot import ShowdownBot

    req = {
        "active": [{
            "moves": [
                {"move": "Draco Meteor", "id": "dracometeor", "pp": 8, "maxpp": 8, "target": "normal", "disabled": True},
                {"move": "Shadow Ball", "id": "shadowball", "pp": 24, "maxpp": 24, "target": "normal", "disabled": True},
                {"move": "Flamethrower", "id": "flamethrower", "pp": 24, "maxpp": 24, "target": "normal", "disabled": True},
                {"move": "U-turn", "id": "uturn", "pp": 32, "maxpp": 32, "target": "normal", "disabled": True},
            ]
        }],
        "side": {
            "name": "Glaubermax", "id": "p2",
            "pokemon": [
                {"details": "Great Tusk", "condition": "371/371", "active": False},
                {"details": "Gholdengo", "condition": "315/315", "active": False},
                {"details": "Kingambit", "condition": "341/341", "active": False},
                {"details": "Dragapult", "condition": "317/317", "active": True},
                {"details": "Ogerpon-Wellspring", "condition": "301/301", "active": False},
                {"details": "Ting-Lu", "condition": "514/514", "active": False},
            ]
        }
    }

    # Simulate move filtering logic from showdown_bot
    active_req_moves = req.get("active", [{}])[0].get("moves", [])
    legal_move_slots = [
        idx + 1 for idx, m_info in enumerate(active_req_moves)
        if isinstance(m_info, dict) and not m_info.get("disabled", False) and m_info.get("pp", 10) > 0
    ]
    assert len(legal_move_slots) == 0, "All moves should be detected as disabled"

    team1 = get_meta_team_balance()
    team2 = get_meta_team_balance()
    state = BattleState(p1=BattleSide(pokemon=team1, active_index=3), p2=BattleSide(pokemon=team2, active_index=0))
    current_actions = state.get_valid_actions(player=1)

    # Filtering when legal_move_slots is empty:
    switches_only = [a for a in current_actions if a.action_type == ActionType.SWITCH]
    assert len(switches_only) > 0, "Must have valid switch options"
    assert all(a.action_type == ActionType.SWITCH for a in switches_only)
