"""Cinematic Animated Terminal Battle Engine for Glaubermon Max.

Features:
- ANSI 24-bit TrueColor interface with type-colored badges and styled cards.
- Smooth animated HP depletion bars with real-time color transitions.
- Paced combat announcements with typewriter animations.
- Dramatic Terastallization visual FX.
- Multithreaded AI thinking spinner during Subgame MCTS.
- In-place HUD redraws for a video game feel.
"""

import os
import sys
import time
import random
import threading
import argparse
from typing import Optional, List, Tuple

import torch
import numpy as np

# Enable ANSI terminal mode on Windows
os.system("")

from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, ActionType, Hazard
from glaubermon.core.actions import Action, MoveAction, SwitchAction
from glaubermon.models.set_transformer import GlaubermonMaxNet
from glaubermon.search.evaluators import NeuralEvaluator
from glaubermon.search.subgame_resolver import SubgameResolver, simulate_turn_transition
from glaubermon.scripts.train_alphazero import generate_competitive_battle
from glaubermon.inference.damage_calc import calculate_damage_rolls


# --- ANSI Colors & Styles ---
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_ITALIC = "\033[3m"
C_CLEAR = "\033[2J\033[H"

# Foreground Colors
FG_RED = "\033[91m"
FG_GREEN = "\033[92m"
FG_YELLOW = "\033[93m"
FG_BLUE = "\033[94m"
FG_MAGENTA = "\033[95m"
FG_CYAN = "\033[96m"
FG_WHITE = "\033[97m"

# Type Badges (Background + Foreground)
TYPE_COLORS = {
    PokemonType.NORMAL: "\033[48;5;244;38;5;15m NORMAL \033[0m",
    PokemonType.FIRE: "\033[48;5;196;38;5;15m FIRE \033[0m",
    PokemonType.WATER: "\033[48;5;33;38;5;15m WATER \033[0m",
    PokemonType.GRASS: "\033[48;5;70;38;5;15m GRASS \033[0m",
    PokemonType.ELECTRIC: "\033[48;5;220;38;5;0m ELECTRIC \033[0m",
    PokemonType.ICE: "\033[48;5;117;38;5;0m ICE \033[0m",
    PokemonType.FIGHTING: "\033[48;5;124;38;5;15m FIGHTING \033[0m",
    PokemonType.POISON: "\033[48;5;129;38;5;15m POISON \033[0m",
    PokemonType.GROUND: "\033[48;5;178;38;5;0m GROUND \033[0m",
    PokemonType.FLYING: "\033[48;5;147;38;5;15m FLYING \033[0m",
    PokemonType.PSYCHIC: "\033[48;5;205;38;5;15m PSYCHIC \033[0m",
    PokemonType.BUG: "\033[48;5;106;38;5;15m BUG \033[0m",
    PokemonType.ROCK: "\033[48;5;137;38;5;15m ROCK \033[0m",
    PokemonType.GHOST: "\033[48;5;97;38;5;15m GHOST \033[0m",
    PokemonType.DRAGON: "\033[48;5;61;38;5;15m DRAGON \033[0m",
    PokemonType.STEEL: "\033[48;5;246;38;5;0m STEEL \033[0m",
    PokemonType.DARK: "\033[48;5;238;38;5;15m DARK \033[0m",
    PokemonType.FAIRY: "\033[48;5;217;38;5;0m FAIRY \033[0m",
}


def typewriter(text: str, delay: float = 0.015, end: str = "\n"):
    """Smooth typewriter pacing for combat text."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(end)
    sys.stdout.flush()


def get_type_badge(poke_type: Optional[PokemonType]) -> str:
    if poke_type is None:
        return ""
    return TYPE_COLORS.get(poke_type, f"[{poke_type.value}]")


def get_animated_hp_bar(current_hp: int, max_hp: int, length: int = 24) -> str:
    """Generate color-shifting HP bar."""
    if max_hp <= 0:
        return f"{FG_RED}[{'░' * length}] 0% (0/0){C_RESET}"

    fraction = max(0.0, min(1.0, current_hp / max_hp))
    filled = int(round(fraction * length))
    empty = length - filled

    if fraction > 0.5:
        bar_color = FG_GREEN
    elif fraction > 0.2:
        bar_color = FG_YELLOW
    else:
        bar_color = FG_RED

    bar = f"{bar_color}{'█' * filled}{C_DIM}{'░' * empty}{C_RESET}"
    pct_text = f"{int(fraction * 100)}%"
    return f"[{bar}] {bar_color}{pct_text:4}{C_RESET} ({current_hp}/{max_hp} HP)"


class ThinkingSpinner:
    """Multithreaded animated spinner while GPU calculates Subgame MCTS."""

    def __init__(self, message: str = "Glaubermon Max is computing Nash Subgame Equilibrium"):
        self.message = message
        self.stop_event = threading.Event()
        self.thread = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _spin(self):
        idx = 0
        while not self.stop_event.is_set():
            frame = self.frames[idx % len(self.frames)]
            sys.stdout.write(f"\r  {FG_CYAN}{C_BOLD}{frame}{C_RESET} {self.message}... ")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.08)
        sys.stdout.write("\r" + " " * (len(self.message) + 15) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if self.thread:
            self.thread.join()


def render_battle_screen(state: BattleState, human_side: int = 1):
    """Render full Pokémon Showdown style animated battle HUD."""
    human = state.p1 if human_side == 1 else state.p2
    ai = state.p2 if human_side == 1 else state.p1

    ai_mon = ai.active_pokemon
    hu_mon = human.active_pokemon

    # Clear screen for seamless redrawing
    sys.stdout.write(C_CLEAR)
    sys.stdout.flush()

    print(f"{FG_CYAN}{C_BOLD}╔{'═' * 74}╗{C_RESET}")
    title = f"⚔️  POKÉMON SHOWDOWN BATTLE STADIUM  |  TURN {state.turn}  ⚔️"
    print(f"{FG_CYAN}{C_BOLD}║{title.center(74)}║{C_RESET}")
    print(f"{FG_CYAN}{C_BOLD}╚{'═' * 74}╝{C_RESET}\n")

    # 1. Opponent Card (Top Right / Opponent Area)
    print(f"  {FG_RED}{C_BOLD}🔴 OPPONENT: GLAUBERMON MAX (AI){C_RESET}")
    if ai_mon:
        t1_badge = get_type_badge(ai_mon.types[0])
        t2_badge = get_type_badge(ai_mon.types[1]) if ai_mon.types[1] else ""
        tera_badge = f" {FG_MAGENTA}{C_BOLD}✨[TERA: {ai_mon.tera_type.value.upper()}]✨{C_RESET}" if ai_mon.is_terastallized else ""

        print(f"  ┌─ {C_BOLD}{ai_mon.species:18}{C_RESET} {t1_badge} {t2_badge}{tera_badge}")
        print(f"  │  HP: {get_animated_hp_bar(ai_mon.current_hp, ai_mon.max_hp)}")
        if ai_mon.boosts:
            b_list = [f"{k.upper()}: {v:+d}" for k, v in ai_mon.boosts.items() if v != 0]
            if b_list:
                print(f"  │  Boosts: {FG_YELLOW}{', '.join(b_list)}{C_RESET}")
        print("  └────────────────────────────────────────────────────────")

    # Opponent Team Pokeballs
    ai_icons = []
    for p in ai.pokemon:
        if p.is_fainted:
            ai_icons.append(f"{C_DIM}✖ {p.species}{C_RESET}")
        elif p == ai_mon:
            ai_icons.append(f"{FG_GREEN}{C_BOLD}● {p.species}{C_RESET}")
        else:
            ai_icons.append(f"{FG_WHITE}● {p.species}{C_RESET}")
    print(f"     Roster: {' | '.join(ai_icons)}\n")

    print(f"  {C_DIM}{'─' * 74}{C_RESET}\n")

    # 2. Player Card (Bottom Left / Player Area)
    print(f"  {FG_GREEN}{C_BOLD}🟢 YOU: TRAINER{C_RESET}")
    if hu_mon:
        t1_badge = get_type_badge(hu_mon.types[0])
        t2_badge = get_type_badge(hu_mon.types[1]) if hu_mon.types[1] else ""
        tera_badge = f" {FG_MAGENTA}{C_BOLD}✨[TERA: {hu_mon.tera_type.value.upper()}]✨{C_RESET}" if hu_mon.is_terastallized else ""

        print(f"  ┌─ {C_BOLD}{hu_mon.species:18}{C_RESET} {t1_badge} {t2_badge}{tera_badge}")
        print(f"  │  HP: {get_animated_hp_bar(hu_mon.current_hp, hu_mon.max_hp)}")
        if hu_mon.boosts:
            b_list = [f"{k.upper()}: {v:+d}" for k, v in hu_mon.boosts.items() if v != 0]
            if b_list:
                print(f"  │  Boosts: {FG_YELLOW}{', '.join(b_list)}{C_RESET}")
        print("  └────────────────────────────────────────────────────────")

    # Player Team Pokeballs
    hu_icons = []
    for p in human.pokemon:
        if p.is_fainted:
            hu_icons.append(f"{C_DIM}✖ {p.species}{C_RESET}")
        elif p == hu_mon:
            hu_icons.append(f"{FG_GREEN}{C_BOLD}● {p.species}{C_RESET}")
        else:
            hu_icons.append(f"{FG_WHITE}● {p.species}{C_RESET}")
    print(f"     Roster: {' | '.join(hu_icons)}\n")

    print(f"{FG_CYAN}═{'═' * 74}═{C_RESET}")


def get_human_action(state: BattleState, human_side: int = 1) -> Action:
    """Prompt human player for an action with styled buttons."""
    side = state.p1 if human_side == 1 else state.p2
    active = side.active_pokemon

    # Forced switch if fainted
    if active.is_fainted:
        print(f"\n{FG_RED}{C_BOLD}Your active Pokémon fainted! Send out a replacement:{C_RESET}")
        switches = side.available_switches()
        for idx in switches:
            p = side.pokemon[idx]
            t_str = f"({p.types[0].value}" + (f"/{p.types[1].value})" if p.types[1] else ")")
            print(f"  {FG_CYAN}[{idx + 1}]{C_RESET} Send out {C_BOLD}{p.species}{C_RESET} {t_str} - {p.current_hp}/{p.max_hp} HP")
        while True:
            choice = input(f"\n{FG_YELLOW}Choose slot number:{C_RESET} ").strip()
            if choice.isdigit() and (int(choice) - 1) in switches:
                target = int(choice) - 1
                return SwitchAction(target_slot=target + 1, species=side.pokemon[target].species)
            print(f"{FG_RED}Invalid choice. Please select an available bench slot.{C_RESET}")

    # Standard Battle Menu
    print(f"\n{C_BOLD}⚔️  ATTACK MOVES:{C_RESET}")
    for i, m in enumerate(active.moves):
        t_badge = get_type_badge(m.move_type)
        cat_badge = f"{C_BOLD}{m.category.value[:4].upper()}{C_RESET}"
        prio_str = f" (+{m.priority})" if m.priority > 0 else ""
        print(f"  {FG_CYAN}[{i + 1}]{C_RESET} {C_BOLD}{m.name:16}{C_RESET} | {t_badge} | {cat_badge:4} | BP: {m.base_power:<3}{prio_str} | PP: {m.pp}")

    can_tera = not side.is_tera_used and active.tera_type is not None and not active.is_terastallized
    if can_tera:
        print(f"\n{FG_MAGENTA}{C_BOLD}✨ TERASTALLIZATION (Tera Type: {active.tera_type.value.upper()}):{C_RESET}")
        for i, m in enumerate(active.moves):
            print(f"  {FG_MAGENTA}[t{i + 1}]{C_RESET} Terastallize + {m.name}")

    print(f"\n{C_BOLD}🔄 POKÉMON SWITCHES:{C_RESET}")
    switches = side.available_switches()
    for idx in switches:
        p = side.pokemon[idx]
        t_str = f"({p.types[0].value}" + (f"/{p.types[1].value})" if p.types[1] else ")")
        print(f"  {FG_BLUE}[s{idx + 1}]{C_RESET} Switch to {C_BOLD}{p.species:12}{C_RESET} {t_str} ({p.current_hp}/{p.max_hp} HP)")

    while True:
        choice = input(f"\n{FG_YELLOW}Enter your command (1-4, t1-t4, s1-s6):{C_RESET} ").strip().lower()

        # Regular move (1-4)
        if choice in ("1", "2", "3", "4"):
            slot = int(choice) - 1
            if slot < len(active.moves):
                m = active.moves[slot]
                return MoveAction(move_id=m.id, move_slot=slot + 1, is_tera=False)

        # Tera move (t1-t4)
        elif choice in ("t1", "t2", "t3", "t4") and can_tera:
            slot = int(choice[1]) - 1
            if slot < len(active.moves):
                m = active.moves[slot]
                return MoveAction(move_id=m.id, move_slot=slot + 1, is_tera=True, tera_type=active.tera_type)

        # Switch (s1-s6)
        elif choice.startswith("s") and choice[1:].isdigit():
            target = int(choice[1:]) - 1
            if target in switches:
                return SwitchAction(target_slot=target + 1, species=side.pokemon[target].species)

        print(f"{FG_RED}Invalid command. Try 1-4 for moves, t1-t4 for Terastallize, or s1-s6 for switch.{C_RESET}")


def execute_turn_with_animations(state: BattleState, hu_act: Action, ai_act: Action) -> BattleState:
    """Simulate turn with dramatic announcements and visual HP animations."""
    print(f"\n{FG_YELLOW}--- TURN EXECUTION ---{C_RESET}")
    time.sleep(0.3)

    # 1. Announcements
    # Switches
    if hu_act.action_type == ActionType.SWITCH:
        target_sp = getattr(hu_act, "species", "bench Pokémon")
        typewriter(f"🔄 Trainer withdrew {state.p1.active_pokemon.species} and sent out {FG_GREEN}{C_BOLD}{target_sp}{C_RESET}!")
        time.sleep(0.4)

    if ai_act.action_type == ActionType.SWITCH:
        target_sp = getattr(ai_act, "species", "bench Pokémon")
        typewriter(f"🔄 Glaubermon Max withdrew {state.p2.active_pokemon.species} and sent out {FG_RED}{C_BOLD}{target_sp}{C_RESET}!")
        time.sleep(0.4)

    # Terastallize
    if getattr(hu_act, "is_tera", False):
        tera_t = getattr(hu_act, "tera_type", PokemonType.NORMAL).value.upper()
        typewriter(f"\n{FG_MAGENTA}{C_BOLD}✨✨✨ TRAINER'S {state.p1.active_pokemon.species} TERASTALLIZED INTO {tera_t} TYPE! ✨✨✨{C_RESET}\n")
        time.sleep(0.6)

    if getattr(ai_act, "is_tera", False):
        tera_t = getattr(ai_act, "tera_type", PokemonType.NORMAL).value.upper()
        typewriter(f"\n{FG_MAGENTA}{C_BOLD}✨✨✨ GLAUBERMON MAX'S {state.p2.active_pokemon.species} TERASTALLIZED INTO {tera_t} TYPE! ✨✨✨{C_RESET}\n")
        time.sleep(0.6)

    # Attack moves
    p1_active = state.p1.active_pokemon
    p2_active = state.p2.active_pokemon

    # Run transition simulation
    new_state = simulate_turn_transition(state, hu_act, ai_act)

    # Report Move 1
    if hu_act.action_type == ActionType.MOVE and p1_active and not p1_active.is_fainted:
        slot = getattr(hu_act, "move_slot") - 1
        m_name = p1_active.moves[slot].name
        typewriter(f"💥 {FG_GREEN}{p1_active.species}{C_RESET} used {C_BOLD}{m_name}{C_RESET}!")
        time.sleep(0.3)

    # Report Move 2
    if ai_act.action_type == ActionType.MOVE and p2_active and not p2_active.is_fainted:
        slot = getattr(ai_act, "move_slot") - 1
        m_name = p2_active.moves[slot].name
        typewriter(f"💥 {FG_RED}{p2_active.species}{C_RESET} used {C_BOLD}{m_name}{C_RESET}!")
        time.sleep(0.3)

    # Check Faints
    if p1_active.is_fainted:
        typewriter(f"\n💀 {FG_RED}{p1_active.species} fainted!{C_RESET}")
        time.sleep(0.5)

    if p2_active.is_fainted:
        typewriter(f"\n💀 {FG_GREEN}Opposing {p2_active.species} fainted!{C_RESET}")
        time.sleep(0.5)

    time.sleep(0.8)
    return new_state


from glaubermon.data.meta_teams import create_meta_battle, META_TEAMS


def play_cinematic_battle(depth: int = 2, team: str = "balance", checkpoint_path: str = "checkpoints/glaubermon_alphazero_latest.pt"):
    """Main animated battle loop with 6v6 tournament meta teams."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{FG_CYAN}{C_BOLD}Initializing Glaubermon Max on {device}...{C_RESET}")

    model = GlaubermonMaxNet(d_model=256, nhead=8, num_actions=14).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"{FG_GREEN}Loaded AlphaZero Model from: {checkpoint_path}{C_RESET}")
    elif os.path.exists("checkpoints/glaubermon_max_latest.pt"):
        model.load_state_dict(torch.load("checkpoints/glaubermon_max_latest.pt", map_location=device))
        print(f"{FG_GREEN}Loaded Grandmaster Model from: checkpoints/glaubermon_max_latest.pt{C_RESET}")

    evaluator = NeuralEvaluator(model, device)
    resolver = SubgameResolver(evaluator=evaluator)

    state = create_meta_battle(team, team)
    typewriter(f"\n{FG_CYAN}{C_BOLD}6v6 Tournament Match Initialized! Meta Archetype: {team.upper()}{C_RESET}\n", delay=0.01)
    time.sleep(1.0)

    while not state.is_game_over and state.turn <= 50:
        render_battle_screen(state, human_side=1)

        # 1. Human Action
        hu_act = get_human_action(state, human_side=1)

        # 2. AI Action with Animated Spinner
        inv_state = BattleState(
            p1=state.p2,
            p2=state.p1,
            weather=state.weather,
            terrain=state.terrain,
            turn=state.turn
        )
        with ThinkingSpinner(f"Glaubermon Max calculating Depth {depth} Nash Subgame"):
            ai_act, ai_strat, ai_actions, eval_val = resolver.resolve_turn(inv_state, depth=depth, sample=False)

        # Print AI Nash Thought Analysis
        print(f"\n{FG_MAGENTA}{C_BOLD}--- Glaubermon Max Nash Strategy Inspector ---{C_RESET}")
        for act, prob in sorted(zip(ai_actions, ai_strat), key=lambda x: x[1], reverse=True):
            if prob > 0.02:
                print(f"  • {C_BOLD}{act.to_showdown_command()}{C_RESET}: {prob * 100:.1f}%")
        adv = f"{FG_RED}AI Advantage{C_RESET}" if eval_val > 0.1 else (f"{FG_GREEN}Trainer Advantage{C_RESET}" if eval_val < -0.1 else f"{FG_YELLOW}Even{C_RESET}")
        print(f"  Position Evaluation: {C_BOLD}{eval_val:+.3f}{C_RESET} ({adv})")

        # 3. Animate the turn!
        state = execute_turn_with_animations(state, hu_act, ai_act)

    render_battle_screen(state, human_side=1)
    if state.winner == 1:
        print(f"\n{FG_GREEN}{C_BOLD}🏆 VICTORY! You defeated Glaubermon Max in {state.turn} turns! 🏆{C_RESET}\n")
    elif state.winner == 2:
        print(f"\n{FG_RED}{C_BOLD}💀 DEFEAT! Glaubermon Max triumphed in {state.turn} turns! 💀{C_RESET}\n")
    else:
        print(f"\n{FG_YELLOW}{C_BOLD}🤝 DRAW! Reached turn limit.{C_RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cinematic Battle against Glaubermon Max")
    parser.add_argument("--depth", type=int, default=2, help="Subgame lookahead search depth (1 or 2)")
    parser.add_argument("--team", type=str, default="balance", choices=["balance", "hyper_offense", "stall"], help="Tournament meta team archetype")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/glaubermon_alphazero_latest.pt")
    args = parser.parse_args()

    play_cinematic_battle(depth=args.depth, team=args.team, checkpoint_path=args.checkpoint)
