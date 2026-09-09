"""Unit tests for protocol log deductions and Bayesian particle filtering."""

import pytest
from glaubermon.core.types import PokemonType
from glaubermon.inference.log_deducer import LogDeducer
from glaubermon.inference.belief_state import BeliefStateTracker, PokemonCandidateBuild


def test_boots_exclusion_from_hazard_damage():
    deducer = LogDeducer()
    # Simulate receiving Stealth Rock damage
    line = "|-damage|p2a: Dragonite|75/100|[from] Stealth Rock"
    deducer.parse_line(line)

    ded = deducer.get_or_create("dragonite")
    assert "heavydutyboots" in ded.excluded_items, "Dragonite should exclude Heavy-Duty Boots"


def test_choice_lock_exclusion():
    deducer = LogDeducer()
    # Turn 1: Kingambit uses Sucker Punch
    deducer.parse_line("|move|p2a: Kingambit|Sucker Punch|p1a: Dragapult")
    ded = deducer.get_or_create("kingambit")
    assert ded.choice_locked_move == "suckerpunch"

    # Turn 2: Kingambit uses Iron Head without switching
    deducer.parse_line("|move|p2a: Kingambit|Iron Head|p1a: Dragapult")
    assert "choicescarf" in ded.excluded_items
    assert "choiceband" in ded.excluded_items


def test_particle_filtering_with_deductions():
    tracker = BeliefStateTracker(num_particles_per_mon=3)
    priors = [
        PokemonCandidateBuild(
            moves=["earthquake", "stealthrock"],
            item="heavydutyboots",
            ability="unaware",
            ev_spread={},
            tera_type=PokemonType.WATER,
            weight=1.0
        ),
        PokemonCandidateBuild(
            moves=["earthquake", "liquidation"],
            item="leftovers",
            ability="unaware",
            ev_spread={},
            tera_type=PokemonType.FAIRY,
            weight=2.0
        )
    ]
    tracker.initialize_from_priors("clodsire", priors)

    deducer = LogDeducer()
    deducer.parse_line("|-damage|p2a: Clodsire|88/100|[from] Stealth Rock")
    ded = deducer.get_or_create("clodsire")

    tracker.filter_with_deductions(ded)
    candidates = tracker.particles["clodsire"]

    # Candidate with heavy-duty boots should be filtered out!
    assert len(candidates) == 1
    assert candidates[0].item == "leftovers"
