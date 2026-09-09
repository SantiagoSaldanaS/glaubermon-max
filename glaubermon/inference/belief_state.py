"""Bayesian Particle Filter for tracking opponent belief states."""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory
from glaubermon.inference.log_deducer import OpponentDeductions


@dataclass
class PokemonCandidateBuild:
    """A candidate build for an opponent Pokémon."""
    moves: List[str]
    item: str
    ability: str
    ev_spread: Dict[str, int]
    tera_type: PokemonType
    weight: float = 1.0


class BeliefStateTracker:
    """Maintains probabilistic belief distributions over opponent hidden sets."""

    def __init__(self, num_particles_per_mon: int = 50):
        self.num_particles = num_particles_per_mon
        # species -> list of candidate builds
        self.particles: Dict[str, List[PokemonCandidateBuild]] = {}

    def initialize_from_priors(self, species: str, prior_builds: List[PokemonCandidateBuild]):
        """Seed the particle filter with prior distributions (e.g., from Smogon stats)."""
        clean = species.strip().lower()
        if not prior_builds:
            # Fallback default build
            prior_builds = [
                PokemonCandidateBuild(
                    moves=["earthquake", "closecombat", "rapidspin", "iceplanner"],
                    item="boosterenergy",
                    ability="protosynthesis",
                    ev_spread={"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252},
                    tera_type=PokemonType.ICE,
                    weight=1.0
                )
            ]
        self.particles[clean] = list(prior_builds)

    def filter_with_deductions(self, deductions: OpponentDeductions):
        """Eliminate impossible particles based on deterministic deductions."""
        clean = deductions.species
        if clean not in self.particles:
            return

        valid_particles = []
        for p in self.particles[clean]:
            # 1. Must contain all revealed moves
            if not deductions.revealed_moves.issubset(set(p.moves)):
                continue

            # 2. Must not contain excluded items
            if p.item.lower() in deductions.excluded_items:
                continue

            # 3. If item is revealed, must match
            if deductions.revealed_item and p.item.lower() != deductions.revealed_item:
                continue

            # 4. If ability is revealed, must match
            if deductions.revealed_ability and p.ability.lower() != deductions.revealed_ability:
                continue

            valid_particles.append(p)

        if valid_particles:
            self.particles[clean] = valid_particles
        else:
            # Re-seed if evidence pruned everything (prevent empty particle set)
            repaired = PokemonCandidateBuild(
                moves=list(deductions.revealed_moves) if deductions.revealed_moves else ["tackle"],
                item=deductions.revealed_item or "leftovers",
                ability=deductions.revealed_ability or "pressure",
                ev_spread={"hp": 252, "atk": 0, "def": 128, "spa": 0, "spd": 128, "spe": 0},
                tera_type=PokemonType.NORMAL,
                weight=1.0
            )
            self.particles[clean] = [repaired]

    def sample_candidate_build(self, species: str) -> PokemonCandidateBuild:
        """Sample a candidate build weighted by likelihood."""
        clean = species.strip().lower()
        if clean not in self.particles or not self.particles[clean]:
            return PokemonCandidateBuild(
                moves=["earthquake", "closecombat", "stealthrock", "rapidspin"],
                item="leftovers",
                ability="unaware",
                ev_spread={"hp": 252, "atk": 0, "def": 252, "spa": 0, "spd": 4, "spe": 0},
                tera_type=PokemonType.WATER
            )

        candidates = self.particles[clean]
        weights = [c.weight for c in candidates]
        total_w = sum(weights)
        if total_w <= 0:
            return random.choice(candidates)
        return random.choices(candidates, weights=weights, k=1)[0]
