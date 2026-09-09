"""Unit tests verifying:
1. All available bench switches (including slots 4 and 5) are preserved in SubgameResolver.
2. Team preview lead selection avoids weak matchups (Ting-Lu vs Great Tusk / Iron Valiant)
   and selects strong counter leads (Ogerpon-Wellspring / Dragapult / Gholdengo).
"""

import pytest
from glaubermon.data.meta_teams import create_meta_battle
from glaubermon.search.subgame_resolver import SubgameResolver
from glaubermon.search.evaluators import HeuristicEvaluator
from glaubermon.core.actions import ActionType
from glaubermon.client.showdown_bot import ShowdownBot


def test_subgame_preserves_all_available_bench_switches():
    """Verify that SubgameResolver evaluates all 5 bench switch options without blind array slicing."""
    state = create_meta_battle("balance", "balance")
    # Set Ting-Lu active (index 5)
    state.p1.active_index = 5
    state.p2.active_index = 0

    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    _, _, actions, _ = resolver.resolve_turn(state, depth=1, sample=False)

    switches = [a for a in actions if a.action_type == ActionType.SWITCH]
    switch_species = [s.species for s in switches]

    # Must contain all 5 bench Pokémon
    assert len(switches) == 5, f"Expected 5 switches, got {len(switches)}: {switch_species}"
    assert "Dragapult" in switch_species, "Dragapult (slot 4) was improperly pruned"
    assert "Ogerpon-Wellspring" in switch_species, "Ogerpon-Wellspring (slot 5) was improperly pruned"
    assert "Great Tusk" in switch_species
    assert "Gholdengo" in switch_species
    assert "Kingambit" in switch_species


def test_showdown_bot_select_lead_chinema():
    """Verify team preview lead selection against Chinema's team avoids Ting-Lu and favors Ogerpon/Dragapult."""
    bot = ShowdownBot.__new__(ShowdownBot)
    bot.opp_team = {
        "battle-chinema": ["Great Tusk", "Gholdengo", "Kingambit", "Dragapult", "Ogerpon-Wellspring", "Ting-Lu"]
    }
    order = bot.select_lead_order("battle-chinema")
    lead_slot = int(order[0])

    # Slot 5 is Ogerpon-Wellspring, Slot 4 is Dragapult
    assert lead_slot in (4, 5), f"Expected Ogerpon (5) or Dragapult (4) lead against Great Tusk, got slot {lead_slot}"
    assert lead_slot != 6, "Ting-Lu must not lead into Great Tusk (weak to Close Combat)"
    assert lead_slot != 3, "Kingambit must never lead"


def test_showdown_bot_select_lead_pelol94():
    """Verify team preview lead selection against Pelol94's team avoids Ting-Lu and favors Ogerpon/Gholdengo."""
    bot = ShowdownBot.__new__(ShowdownBot)
    bot.opp_team = {
        "battle-pelol": ["Gliscor", "Great Tusk", "Kingambit", "Ting-Lu", "Iron Valiant", "Dragapult"]
    }
    order = bot.select_lead_order("battle-pelol")
    lead_slot = int(order[0])

    # Slot 5 is Ogerpon, Slot 2 is Gholdengo (walls Gliscor and Valiant)
    assert lead_slot in (2, 4, 5), f"Expected Ogerpon (5), Gholdengo (2), or Dragapult (4), got slot {lead_slot}"
    assert lead_slot != 6, "Ting-Lu must not lead into Great Tusk / Iron Valiant"
    assert lead_slot != 3, "Kingambit must never lead"
