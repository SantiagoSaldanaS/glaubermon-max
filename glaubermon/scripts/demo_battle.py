"""Interactive demonstration battle: Glaubermon Max (Nash Subgame Search + Trained Neural Net) vs Baseline."""

import os
import torch
import numpy as np
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, ActionType, Hazard
from glaubermon.models.set_transformer import GlaubermonMaxNet
from glaubermon.search.evaluators import HeuristicEvaluator, NeuralEvaluator
from glaubermon.search.subgame_resolver import SubgameResolver, simulate_turn_transition
from glaubermon.inference.damage_calc import calculate_damage_rolls


def create_sample_teams() -> BattleState:
    # --- Glaubermon Max Team ---
    p1_mons = [
        Pokemon(
            species="Great Tusk",
            types=(PokemonType.GROUND, PokemonType.FIGHTING),
            max_hp=371, current_hp=371,
            moves=[
                Move.create("Close Combat", PokemonType.FIGHTING, MoveCategory.PHYSICAL, base_power=120),
                Move.create("Headlong Rush", PokemonType.GROUND, MoveCategory.PHYSICAL, base_power=120),
                Move.create("Ice Spinner", PokemonType.ICE, MoveCategory.PHYSICAL, base_power=80),
                Move.create("Rapid Spin", PokemonType.NORMAL, MoveCategory.PHYSICAL, base_power=50)
            ],
            raw_stats={"hp": 371, "atk": 361, "def": 298, "spa": 127, "spd": 142, "spe": 273},
            tera_type=PokemonType.ICE
        ),
        Pokemon(
            species="Dragapult",
            types=(PokemonType.DRAGON, PokemonType.GHOST),
            max_hp=317, current_hp=317,
            moves=[
                Move.create("Shadow Ball", PokemonType.GHOST, MoveCategory.SPECIAL, base_power=80),
                Move.create("Draco Meteor", PokemonType.DRAGON, MoveCategory.SPECIAL, base_power=130),
                Move.create("U-turn", PokemonType.BUG, MoveCategory.PHYSICAL, base_power=70),
                Move.create("Flamethrower", PokemonType.FIRE, MoveCategory.SPECIAL, base_power=90)
            ],
            raw_stats={"hp": 317, "atk": 257, "def": 186, "spa": 299, "spd": 186, "spe": 421},
            tera_type=PokemonType.GHOST
        ),
        Pokemon(
            species="Kingambit",
            types=(PokemonType.DARK, PokemonType.STEEL),
            max_hp=404, current_hp=404,
            moves=[
                Move.create("Kowtow Cleave", PokemonType.DARK, MoveCategory.PHYSICAL, base_power=85),
                Move.create("Sucker Punch", PokemonType.DARK, MoveCategory.PHYSICAL, base_power=70, priority=1),
                Move.create("Iron Head", PokemonType.STEEL, MoveCategory.PHYSICAL, base_power=80),
                Move.create("Swords Dance", PokemonType.NORMAL, MoveCategory.STATUS, base_power=0)
            ],
            raw_stats={"hp": 404, "atk": 405, "def": 276, "spa": 140, "spd": 206, "spe": 136},
            tera_type=PokemonType.FLYING
        )
    ]

    # --- Baseline Opponent Team ---
    p2_mons = [
        Pokemon(
            species="Gholdengo",
            types=(PokemonType.STEEL, PokemonType.GHOST),
            max_hp=315, current_hp=315,
            moves=[
                Move.create("Make It Rain", PokemonType.STEEL, MoveCategory.SPECIAL, base_power=120),
                Move.create("Shadow Ball", PokemonType.GHOST, MoveCategory.SPECIAL, base_power=80),
                Move.create("Focus Blast", PokemonType.FIGHTING, MoveCategory.SPECIAL, base_power=120),
                Move.create("Nasty Plot", PokemonType.DARK, MoveCategory.STATUS, base_power=0)
            ],
            raw_stats={"hp": 315, "atk": 140, "def": 226, "spa": 399, "spd": 218, "spe": 267},
            tera_type=PokemonType.FIGHTING
        ),
        Pokemon(
            species="Dondozo",
            types=(PokemonType.WATER, None),
            max_hp=504, current_hp=504,
            moves=[
                Move.create("Liquidation", PokemonType.WATER, MoveCategory.PHYSICAL, base_power=85),
                Move.create("Body Press", PokemonType.FIGHTING, MoveCategory.PHYSICAL, base_power=80),
                Move.create("Rest", PokemonType.PSYCHIC, MoveCategory.STATUS, base_power=0),
                Move.create("Sleep Talk", PokemonType.NORMAL, MoveCategory.STATUS, base_power=0)
            ],
            raw_stats={"hp": 504, "atk": 236, "def": 361, "spa": 149, "spd": 166, "spe": 106},
            tera_type=PokemonType.GRASS
        ),
        Pokemon(
            species="Iron Valiant",
            types=(PokemonType.FAIRY, PokemonType.FIGHTING),
            max_hp=289, current_hp=289,
            moves=[
                Move.create("Moonblast", PokemonType.FAIRY, MoveCategory.SPECIAL, base_power=95),
                Move.create("Close Combat", PokemonType.FIGHTING, MoveCategory.PHYSICAL, base_power=120),
                Move.create("Knock Off", PokemonType.DARK, MoveCategory.PHYSICAL, base_power=65),
                Move.create("Thunderbolt", PokemonType.ELECTRIC, MoveCategory.SPECIAL, base_power=90)
            ],
            raw_stats={"hp": 289, "atk": 296, "def": 216, "spa": 339, "spd": 156, "spe": 364},
            tera_type=PokemonType.FAIRY
        )
    ]

    return BattleState(
        p1=BattleSide(pokemon=p1_mons, hazards={Hazard.STEALTH_ROCK: 1}),
        p2=BattleSide(pokemon=p2_mons, hazards={Hazard.STEALTH_ROCK: 1})
    )


def select_baseline_action(state: BattleState) -> any:
    """Baseline heuristic player (selects move that deals maximum immediate damage)."""
    actions = state.get_valid_actions(player=2)
    p2_active = state.p2.active_pokemon
    p1_active = state.p1.active_pokemon

    if not actions:
        return None

    best_action = actions[0]
    best_dmg = -1

    for a in actions:
        if a.action_type == ActionType.MOVE and p2_active and p1_active:
            slot = getattr(a, "move_slot") - 1
            if 0 <= slot < len(p2_active.moves):
                mv = p2_active.moves[slot]
                rolls = calculate_damage_rolls(p2_active, p1_active, mv)
                avg_dmg = sum(rolls) / len(rolls)
                if avg_dmg > best_dmg:
                    best_dmg = avg_dmg
                    best_action = a
    return best_action


def run_demo_battle(use_neural: bool = True, depth: int = 1):
    print("=" * 70)
    print("  GLAUBERMON MAX: GAME-THEORETIC SIMULTANEOUS SUBGAME RESOLVER DEMO")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/glaubermon_max_latest.pt"

    if use_neural and os.path.exists(ckpt_path):
        print(f"Loading trained neural model from {ckpt_path} on {device}...")
        model = GlaubermonMaxNet(d_model=128, nhead=4).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        evaluator = NeuralEvaluator(model, device)
        print("Using trained GlaubermonMaxNet Set Transformer for position evaluation.")
    else:
        print("Using HeuristicEvaluator.")
        evaluator = HeuristicEvaluator()

    state = create_sample_teams()
    resolver = SubgameResolver(evaluator=evaluator)

    while not state.is_game_over and state.turn <= 25:
        p1_act = state.p1.active_pokemon
        p2_act = state.p2.active_pokemon

        print(f"\n--- Turn {state.turn} ---")
        print(f"P1 (Glaubermon Max): {p1_act.species} ({p1_act.current_hp}/{p1_act.max_hp} HP)")
        print(f"P2 (Baseline):       {p2_act.species} ({p2_act.current_hp}/{p2_act.max_hp} HP)")

        # 1. Glaubermon Max resolves simultaneous Nash Equilibrium
        glaubermon_action, p1_strat, actions, expected_val = resolver.resolve_turn(state, depth=depth, sample=False)

        print("\nGlaubermon Max Nash Strategy Distribution:")
        for act, prob in zip(actions, p1_strat):
            if prob > 0.01:
                desc = getattr(act, 'move_id', getattr(act, 'species', 'Action'))
                tera_str = " (Tera)" if getattr(act, 'is_tera', False) else ""
                print(f"  • {act.action_type.name} [{desc}{tera_str}]: {prob * 100:.1f}%")
        print(f"Position Evaluation: {expected_val:+.3f} (P1 Advantage)")

        # 2. Baseline picks action
        baseline_action = select_baseline_action(state)

        # 3. Simulate turn transition
        prev_p1_species = state.p1.active_pokemon.species
        prev_p2_species = state.p2.active_pokemon.species
        state = simulate_turn_transition(state, glaubermon_action, baseline_action)

        # Check for faints & force switch
        if state.p1.active_pokemon.is_fainted and not state.p1.is_all_fainted:
            fainted_sp = state.p1.active_pokemon.species
            switches = state.p1.available_switches()
            if switches:
                state.p1.active_index = switches[0]
                print(f">> P1 {fainted_sp} fainted! Sent out {state.p1.active_pokemon.species}")

        if state.p2.active_pokemon.is_fainted and not state.p2.is_all_fainted:
            fainted_sp = state.p2.active_pokemon.species
            switches = state.p2.available_switches()
            if switches:
                state.p2.active_index = switches[0]
                print(f">> P2 {fainted_sp} fainted! Sent out {state.p2.active_pokemon.species}")

    print("\n" + "=" * 70)
    if state.winner == 1:
        print("  BATTLE RESULT: GLAUBERMON MAX VICTORIOUS!")
    elif state.winner == 2:
        print("  BATTLE RESULT: BASELINE VICTORIOUS!")
    else:
        print("  BATTLE RESULT: DRAW (Turn limit reached)")
    print("=" * 70)


if __name__ == "__main__":
    run_demo_battle(use_neural=True, depth=1)
