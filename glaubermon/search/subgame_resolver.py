"""Depth-Limited Simultaneous-Move Subgame Search Engine."""

import random
from typing import List, Optional, Tuple
import numpy as np
from glaubermon.core.battle_state import BattleState
from glaubermon.core.pokemon import Pokemon
from glaubermon.core.actions import Action, MoveAction, SwitchAction
from glaubermon.core.types import ActionType, Hazard, PokemonType, MoveCategory, StatusCondition, Terrain
from glaubermon.core.constants import get_type_effectiveness, clean_key
from glaubermon.inference.damage_calc import calculate_damage_rolls, is_contact_move, is_removable_item
from glaubermon.search.evaluators import StateEvaluator, HeuristicEvaluator
from glaubermon.search.matrix_solver import solve_zero_sum_game


def apply_entry_hazards(state: BattleState, side_idx: int, mon_idx: int):
    """Apply hazard damage (Stealth Rock, Spikes) to entering Pokémon."""
    side = state.p1 if side_idx == 1 else state.p2
    if 0 <= mon_idx < len(side.pokemon):
        mon = side.pokemon[mon_idx]
        dmg = mon.calculate_hazard_damage(side.hazards)
        if dmg > 0:
            mon.take_damage(dmg)


def apply_tera_boosts(mon: Optional[Pokemon]):
    """Apply Embody Aspect stat boosts when Ogerpon Terastallizes in Gen 9."""
    if not mon:
        return
    spec_clean = clean_key(mon.species)
    item_clean = clean_key(mon.item)
    if "ogerpon" in spec_clean:
        if "hearthflame" in spec_clean or "hearthflame" in item_clean:
            mon.boosts["atk"] = min(6, mon.boosts.get("atk", 0) + 1)
        elif "cornerstone" in spec_clean or "cornerstone" in item_clean:
            mon.boosts["def"] = min(6, mon.boosts.get("def", 0) + 1)
        elif "wellspring" in spec_clean or "wellspring" in item_clean:
            mon.boosts["spd"] = min(6, mon.boosts.get("spd", 0) + 1)
        else:
            # Teal Mask (standard) grants +1 Speed
            mon.boosts["spe"] = min(6, mon.boosts.get("spe", 0) + 1)


def simulate_turn_transition(
    state: BattleState,
    a1: Action,
    a2: Action,
    tie_winner: Optional[str] = None
) -> BattleState:
    """Simulate a single simultaneous turn transition with priority, switching, and damage."""
    s = state.clone()
    p1_active = s.p1.active_pokemon
    p2_active = s.p2.active_pokemon
    if p1_active:
        p1_active.is_protected = False
    if p2_active:
        p2_active.is_protected = False

    # Phase 1: Handle Switches (Switches have highest priority)
    p1_switched = False
    p2_switched = False

    if a1 is not None and a1.action_type == ActionType.SWITCH:
        old_mon = s.p1.active_pokemon
        if old_mon:
            old_mon.choice_locked_move = None
            if not old_mon.is_fainted and (old_mon.ability or "").lower().replace("-", "").replace(" ", "") == "regenerator":
                old_mon.heal(max(1, old_mon.max_hp // 3))
        target_slot = getattr(a1, "target_slot") - 1
        s.p1.active_index = target_slot
        apply_entry_hazards(s, side_idx=1, mon_idx=target_slot)
        p1_active = s.p1.active_pokemon
        p1_switched = True

    if a2 is not None and a2.action_type == ActionType.SWITCH:
        old_mon = s.p2.active_pokemon
        if old_mon:
            old_mon.choice_locked_move = None
            if not old_mon.is_fainted and (old_mon.ability or "").lower().replace("-", "").replace(" ", "") == "regenerator":
                old_mon.heal(max(1, old_mon.max_hp // 3))
        target_slot = getattr(a2, "target_slot") - 1
        s.p2.active_index = target_slot
        apply_entry_hazards(s, side_idx=2, mon_idx=target_slot)
        p2_active = s.p2.active_pokemon
        p2_switched = True

    # Phase 2: Handle Terastallization
    if a1 is not None and a1.action_type == ActionType.MOVE and getattr(a1, "is_tera", False) and not s.p1.is_tera_used:
        if p1_active and not p1_active.is_fainted:
            p1_active.is_terastallized = True
            s.p1.is_tera_used = True
            apply_tera_boosts(p1_active)

    if a2 is not None and a2.action_type == ActionType.MOVE and getattr(a2, "is_tera", False) and not s.p2.is_tera_used:
        if p2_active and not p2_active.is_fainted:
            p2_active.is_terastallized = True
            s.p2.is_tera_used = True
            apply_tera_boosts(p2_active)

    # Phase 3: Handle Moves
    m1 = None
    m2 = None
    if a1 is not None and a1.action_type == ActionType.MOVE and p1_active and not p1_active.is_fainted:
        slot = getattr(a1, "move_slot") - 1
        if 0 <= slot < len(p1_active.moves):
            m1 = p1_active.moves[slot]

    if a2 is not None and a2.action_type == ActionType.MOVE and p2_active and not p2_active.is_fainted:
        slot = getattr(a2, "move_slot") - 1
        if 0 <= slot < len(p2_active.moves):
            m2 = p2_active.moves[slot]

    # Determine move order (Priority first, then Effective Speed)
    if m1 and m2:
        prio1 = m1.priority
        prio2 = m2.priority
        spe1 = p1_active.effective_stat("spe")
        spe2 = p2_active.effective_stat("spe")

        if prio1 > prio2:
            order = [("p1", m1), ("p2", m2)]
        elif prio2 > prio1:
            order = [("p2", m2), ("p1", m1)]
        elif spe1 > spe2:
            order = [("p1", m1), ("p2", m2)]
        elif spe2 > spe1:
            order = [("p2", m2), ("p1", m1)]
        else:
            # Gen 9 Speed tie: coin flip
            if tie_winner == "p2":
                order = [("p2", m2), ("p1", m1)]
            else:
                order = [("p1", m1), ("p2", m2)]
    elif m1:
        order = [("p1", m1)]
    elif m2:
        order = [("p2", m2)]
    else:
        order = []

    # Execute moves in resolved order
    moved_players = set()
    for player, move in order:
        attacker = p1_active if player == "p1" else p2_active
        defender = p2_active if player == "p1" else p1_active

        if attacker and not attacker.is_fainted and defender and not defender.is_fainted:
            m_id = move.id.lower().replace(" ", "").replace("-", "")

            # Lock Choice items to move executed
            atk_it = clean_key(attacker.item)
            if atk_it in ("choicespecs", "choiceband", "choicescarf") and not attacker.choice_locked_move:
                attacker.choice_locked_move = m_id

            # Check if attacker is asleep
            if attacker.status == StatusCondition.SLEEP:
                if m_id == "sleeptalk":
                    other_moves = [m for m in attacker.moves if m.id.lower().replace(" ", "").replace("-", "") != "sleeptalk"]
                    if other_moves:
                        move = random.choice(other_moves)
                        m_id = move.id.lower().replace(" ", "").replace("-", "")
                else:
                    attacker.status_turns -= 1
                    if attacker.status_turns <= 0:
                        attacker.status = StatusCondition.NONE
                    moved_players.add(player)
                    continue  # Fast asleep, cannot move!

            # 0. Sucker Punch failure check:
            # Fails if opponent switched, or opponent used a status move, or opponent already moved!
            if m_id == "suckerpunch":
                opp_move = m2 if player == "p1" else m1
                opp_player = "p2" if player == "p1" else "p1"
                if opp_move is None or opp_move.category == MoveCategory.STATUS or (opp_player in moved_players):
                    moved_players.add(player)
                    continue  # Sucker Punch failed!

            # 1. Protection moves
            if getattr(move, "is_protect", False) or m_id in ("protect", "spikyshield", "detect", "banefulbunker", "silktrap"):
                protect_count = getattr(attacker, "protect_streak", 0)
                # Authentic Gen 9 Protect success decay formula: 1 / 3^k
                success_rate = 1.0 / (3.0 ** protect_count)
                attacker.protect_success_rate = success_rate
                attacker.protect_streak = protect_count + 1
                attacker.last_protect_move = m_id
                moved_players.add(player)
                continue
            else:
                attacker.protect_streak = 0
                attacker.protect_success_rate = 0.0

            actual_dmg = 0
            # 2. Damage calculation
            if move.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                atk_side = s.p1 if player == "p1" else s.p2
                rolls = calculate_damage_rolls(
                    attacker, defender, move, s.weather, s.terrain,
                    attacker_side=atk_side
                )
                median_dmg = rolls[len(rolls) // 2]
                succ_rate = getattr(defender, "protect_success_rate", 0.0)
                is_contact = getattr(move, "is_contact", False) or is_contact_move(m_id, move.category)

                if succ_rate > 0.0:
                    # Defender takes damage proportional to protect failure rate
                    actual_dmg = int(median_dmg * (1.0 - succ_rate))
                    defender.take_damage(actual_dmg)
                    # Spiky shield chip on physical/special contact when protected
                    if succ_rate > 0.5 and getattr(defender, "last_protect_move", "") == "spikyshield" and is_contact:
                        attacker.take_damage(max(1, attacker.max_hp // 8))
                else:
                    actual_dmg = median_dmg
                    defender.take_damage(median_dmg)
                    # Rocky Helmet chip on contact when attack connects
                    def_it = clean_key(defender.item)
                    if def_it == "rockyhelmet" and is_contact:
                        attacker.take_damage(max(1, attacker.max_hp // 6))

                # Pop Air Balloon on direct damage
                if actual_dmg > 0 and clean_key(defender.item) == "airballoon":
                    defender.item = None

                # Knock Off item removal (if attack lands and defender not protected)
                if m_id == "knockoff" and succ_rate <= 0.5 and defender.item:
                    if is_removable_item(defender.item):
                        defender.item = None

                # Drain moves: heal attacker by exact canonical ratio or default 50%
                if getattr(move, "drain", None):
                    num, den = move.drain
                    drain_heal = max(1, actual_dmg * num // den)
                    attacker.heal(drain_heal)
                elif m_id in ("hornleech", "drainpunch", "gigadrain", "absorb", "megadrain", "drainingkiss", "oblivionwing", "bitterblade", "paraboliccharge"):
                    drain_heal = max(1, actual_dmg // 2)
                    attacker.heal(drain_heal)

                # Recoil moves: recoil by exact canonical ratio or standard fraction
                if getattr(move, "recoil", None):
                    num, den = move.recoil
                    attacker.take_damage(max(1, actual_dmg * num // den))
                elif m_id in ("bravebird", "flareblitz", "woodhammer", "wavecrash", "doubleedge"):
                    attacker.take_damage(max(1, actual_dmg // 3))
                elif m_id in ("headsmash",):
                    attacker.take_damage(max(1, actual_dmg // 2))

                # Life Orb recoil (10% max HP)
                if clean_key(attacker.item) == "lifeorb":
                    attacker.take_damage(max(1, attacker.max_hp // 10))

            # 3. Status, Hazard & Boost effects
            def_ab = clean_key(defender.ability)
            is_magic_bounce = (def_ab == "magicbounce" and (getattr(move, "is_reflectable", False) or m_id in ("stealthrock", "spikes", "willowisp", "toxic", "thunderwave", "whirlwind", "roar", "taunt")))

            # Data-driven self-boosts (Swords Dance, Calm Mind, Close Combat, Shell Smash, Rapid Spin, Make It Rain, Draco Meteor, etc.)
            # Damaging attacks only apply self-boosts if not blocked by Protect and not immune
            if getattr(move, "self_boosts", None):
                is_blocked = (succ_rate > 0.5 or actual_dmg == 0) if move.category != MoveCategory.STATUS else False
                if not is_blocked:
                    for stat_k, boost_val in move.self_boosts.items():
                        curr_b = attacker.boosts.get(stat_k, 0)
                        attacker.boosts[stat_k] = max(-6, min(6, curr_b + boost_val))

            # Data-driven target-boosts / stat drops (Screech, Charm, Chilling Water, Mystical Fire, Icy Wind, etc.)
            if getattr(move, "boosts", None):
                is_blocked = (succ_rate > 0.5) or (def_ab == "goodasgold" and any(v < 0 for v in move.boosts.values()))
                if not is_blocked:
                    for stat_k, boost_val in move.boosts.items():
                        curr_b = defender.boosts.get(stat_k, 0)
                        defender.boosts[stat_k] = max(-6, min(6, curr_b + boost_val))

            # Additional field / status mechanics
            if m_id in ("rapidspin", "mortalspin", "tidyup"):
                is_blocked = (succ_rate > 0.5 or actual_dmg == 0) if m_id != "tidyup" else False
                if not is_blocked:
                    user_side = s.p1 if player == "p1" else s.p2
                    user_side.hazards.clear()
                    if m_id == "mortalspin":
                        if PokemonType.POISON not in defender.active_types and PokemonType.STEEL not in defender.active_types and def_ab != "goodasgold":
                            defender.status = StatusCondition.POISON
                elif m_id == "tidyup":
                    attacker.boosts["atk"] = min(6, attacker.boosts.get("atk", 0) + 1)
                    attacker.boosts["spe"] = min(6, attacker.boosts.get("spe", 0) + 1)
            elif m_id == "courtchange":
                s.p1.hazards, s.p2.hazards = s.p2.hazards, s.p1.hazards
            elif m_id == "icespinner":
                s.terrain = Terrain.NONE
            elif m_id == "ceaselessedge":
                opp_side = s.p2 if player == "p1" else s.p1
                opp_side.hazards[Hazard.SPIKES_1] = min(3, opp_side.hazards.get(Hazard.SPIKES_1, 0) + 1)
            elif m_id == "stoneaxe":
                opp_side = s.p2 if player == "p1" else s.p1
                opp_side.hazards[Hazard.STEALTH_ROCK] = 1
            elif m_id == "stealthrock":
                target_side = (s.p1 if player == "p1" else s.p2) if is_magic_bounce else (s.p2 if player == "p1" else s.p1)
                target_side.hazards[Hazard.STEALTH_ROCK] = 1
            elif m_id == "spikes":
                target_side = (s.p1 if player == "p1" else s.p2) if is_magic_bounce else (s.p2 if player == "p1" else s.p1)
                target_side.hazards[Hazard.SPIKES_1] = min(3, target_side.hazards.get(Hazard.SPIKES_1, 0) + 1)
            elif m_id == "defog":
                s.p1.hazards.clear()
                s.p2.hazards.clear()
            elif m_id in ("uturn", "voltswitch", "flipturn"):
                user_side = s.p1 if player == "p1" else s.p2
                succ_rate = getattr(defender, "protect_success_rate", 0.0)
                # If Protect / Spiky Shield succeeded, pivot move is blocked and DOES NOT switch!
                if succ_rate <= 0.5 and not attacker.is_fainted:
                    if clean_key(attacker.ability) == "regenerator":
                        attacker.heal(max(1, attacker.max_hp // 3))
                    switches = user_side.available_switches()
                    if switches:
                        best_sw = switches[0]
                        best_score = -999.0
                        def_types = defender.active_types if defender else (PokemonType.NORMAL, None)
                        for sw_idx in switches:
                            sw_mon = user_side.pokemon[sw_idx]
                            type_weakness = 1.0
                            for dt in def_types:
                                if dt:
                                    t1, t2 = sw_mon.active_types
                                    type_weakness *= get_type_effectiveness(dt, t1, t2)
                            score = sw_mon.hp_percent * 2.0 - type_weakness
                            if score > best_score:
                                best_score = score
                                best_sw = sw_idx

                        user_side.active_index = best_sw
                        apply_entry_hazards(s, side_idx=1 if player == "p1" else 2, mon_idx=best_sw)
                        if player == "p1":
                            p1_active = s.p1.active_pokemon
                        else:
                            p2_active = s.p2.active_pokemon
            elif m_id in ("whirlwind", "roar", "dragontail", "circlethrow"):
                if m_id in ("whirlwind", "roar") and def_ab == "goodasgold":
                    pass  # Good as Gold is immune to status phazing
                elif is_magic_bounce and m_id in ("whirlwind", "roar"):
                    # Magic Bounce reflects phazing back to user side
                    u_side = s.p1 if player == "p1" else s.p2
                    u_sw = u_side.available_switches()
                    if u_sw:
                        u_side.active_index = u_sw[0]
                        apply_entry_hazards(s, side_idx=1 if player == "p1" else 2, mon_idx=u_sw[0])
                        if player == "p1":
                            p1_active = s.p1.active_pokemon
                        else:
                            p2_active = s.p2.active_pokemon
                else:
                    opp_side = s.p2 if player == "p1" else s.p1
                    if defender:
                        defender.boosts = {k: 0 for k in defender.boosts}
                    opp_sw = opp_side.available_switches()
                    if opp_sw:
                        opp_target = opp_sw[0]
                        opp_side.active_index = opp_target
                        apply_entry_hazards(s, side_idx=2 if player == "p1" else 1, mon_idx=opp_target)
                        if player == "p1":
                            p2_active = s.p2.active_pokemon
                        else:
                            p1_active = s.p1.active_pokemon
            elif (getattr(move, "is_heal", False) and move.category == MoveCategory.STATUS) or m_id in ("recover", "roost", "slackoff", "softboiled", "wish", "synthesis", "moonlight", "morningsun", "shoreup", "milkdrink", "healorder"):
                if attacker.current_hp < attacker.max_hp:
                    attacker.heal(attacker.max_hp // 2)
            elif m_id == "rest":
                if attacker.current_hp < attacker.max_hp and attacker.status != StatusCondition.SLEEP:
                    attacker.heal(attacker.max_hp)
                    attacker.status = StatusCondition.SLEEP
                    attacker.status_turns = 2
            elif m_id == "willowisp":
                target_mon = attacker if is_magic_bounce else defender
                t_ab = clean_key(target_mon.ability)
                if PokemonType.FIRE not in target_mon.active_types and t_ab != "goodasgold":
                    target_mon.status = StatusCondition.BURN
            elif m_id == "toxic":
                target_mon = attacker if is_magic_bounce else defender
                t_ab = clean_key(target_mon.ability)
                if PokemonType.POISON not in target_mon.active_types and PokemonType.STEEL not in target_mon.active_types and t_ab != "goodasgold":
                    target_mon.status = StatusCondition.TOXIC
            elif m_id == "thunderwave":
                target_mon = attacker if is_magic_bounce else defender
                t_ab = clean_key(target_mon.ability)
                if PokemonType.ELECTRIC not in target_mon.active_types and PokemonType.GROUND not in target_mon.active_types and t_ab != "goodasgold":
                    target_mon.status = StatusCondition.PARALYSIS

            moved_players.add(player)

    # Phase 4: End-of-turn effects (Leftovers, Black Sludge, Poison Heal, Status Orbs, Burn, Poison)
    for act_mon in (s.p1.active_pokemon, s.p2.active_pokemon):
        if act_mon and not act_mon.is_fainted:
            ab = clean_key(act_mon.ability)
            it = clean_key(act_mon.item)

            # Leftovers
            if it == "leftovers":
                act_mon.heal(max(1, act_mon.max_hp // 16))

            # Black Sludge
            if it == "blacksludge":
                if PokemonType.POISON in act_mon.active_types:
                    act_mon.heal(max(1, act_mon.max_hp // 16))
                else:
                    act_mon.take_damage(max(1, act_mon.max_hp // 8))

            # Status Orbs
            if act_mon.status == StatusCondition.NONE:
                if it == "flameorb" and PokemonType.FIRE not in act_mon.active_types:
                    act_mon.status = StatusCondition.BURN
                elif it == "toxicorb" and PokemonType.POISON not in act_mon.active_types and PokemonType.STEEL not in act_mon.active_types:
                    act_mon.status = StatusCondition.TOXIC

            # Poison / Toxic (with Poison Heal support)
            if act_mon.status in (StatusCondition.POISON, StatusCondition.TOXIC):
                if ab == "poisonheal":
                    act_mon.heal(max(1, act_mon.max_hp // 8))
                else:
                    act_mon.take_damage(max(1, act_mon.max_hp // 8))
            elif act_mon.status == StatusCondition.BURN:
                act_mon.take_damage(max(1, act_mon.max_hp // 16))

    s.turn += 1

    # Auto-switch fainted active Pokémon so subsequent turn evaluations are clean
    # Note: Do not inflict entry hazards here; hazards only trigger when a Pokémon actually switches in or is chosen via forced switch
    if s.p1.active_pokemon and s.p1.active_pokemon.is_fainted and not s.p1.is_all_fainted:
        sw1 = s.p1.available_switches()
        if sw1:
            s.p1.active_index = sw1[0]

    if s.p2.active_pokemon and s.p2.active_pokemon.is_fainted and not s.p2.is_all_fainted:
        sw2 = s.p2.available_switches()
        if sw2:
            s.p2.active_index = sw2[0]

    return s


class SubgameResolver:
    """Solves simultaneous extensive-form turns to calculate Nash mixed strategies."""

    def __init__(
        self,
        evaluator: Optional[StateEvaluator] = None,
        switch_penalty: float = 0.25,
        faint_penalty: float = 4.00
    ):
        self.evaluator = evaluator or HeuristicEvaluator()
        self.switch_penalty = switch_penalty
        self.faint_penalty = faint_penalty

    def resolve_turn(
        self,
        state: BattleState,
        depth: int = 1,
        sample: bool = True,
        p1_actions_override: Optional[List[Action]] = None,
        p1_sucker_streak: int = 0,
        last_action_was_switch: bool = False,
        cand_beam: Optional[Tuple[int, int]] = None,
        is_leaf_eval: bool = False
    ) -> Tuple[Action, np.ndarray, List[Action], float]:
        """Compute the Nash Equilibrium mixed strategy for the current turn.

        Returns:
            chosen_action: The selected action (sampled from mixed strategy or argmax).
            p1_strategy: The full Nash probability distribution over p1_actions.
            p1_actions: List of valid actions for P1.
            expected_value: The game-theoretic value of the position.
        """
        if state.is_game_over:
            val = 1.0 if state.winner == 1 else -1.0
            return (None, np.array([]), [], val)

        p1_actions = p1_actions_override if p1_actions_override is not None else state.get_valid_actions(player=1)
        p2_actions = state.get_valid_actions(player=2)

        orig_idx1 = state.p1.active_index
        orig_idx2 = state.p2.active_index
        p1_mon_before = state.p1.pokemon[orig_idx1] if 0 <= orig_idx1 < len(state.p1.pokemon) else None
        p2_mon_before = state.p2.pokemon[orig_idx2] if 0 <= orig_idx2 < len(state.p2.pokemon) else None

        # Leaf Quiescence / Optimal Defensive Action Pruning for Deep Lookahead:
        # At leaf evaluation depth, evaluate all active moves plus the top 2 best defensive counter switches
        # (sorted by type resistance to opponent's attacks and HP) to ensure zero strategic capability loss.
        if is_leaf_eval:
            p1_moves = [a for a in p1_actions if a.action_type == ActionType.MOVE and not getattr(a, "is_tera", False)]
            p1_switches = [a for a in p1_actions if a.action_type == ActionType.SWITCH]
            if p2_mon_before and p1_switches:
                opp_types = [t for t in p2_mon_before.types if t]
                def _sw1_score(sw_act):
                    slot = getattr(sw_act, "target_slot", 1) - 1
                    mon = state.p1.pokemon[slot] if 0 <= slot < len(state.p1.pokemon) else None
                    if not mon or mon.is_fainted:
                        return 999.0
                    t1 = mon.active_types[0]
                    t2 = mon.active_types[1] if len(mon.active_types) > 1 else None
                    worst_eff = max((get_type_effectiveness(ot, t1, t2) for ot in opp_types), default=1.0)
                    return worst_eff - 0.5 * mon.hp_percent
                p1_switches.sort(key=_sw1_score)
            p1_actions = (p1_moves if p1_moves else p1_actions[:1]) + p1_switches[:2]

            p2_moves = [a for a in p2_actions if a.action_type == ActionType.MOVE and not getattr(a, "is_tera", False)]
            p2_switches = [a for a in p2_actions if a.action_type == ActionType.SWITCH]
            if p1_mon_before and p2_switches:
                p1_types = [t for t in p1_mon_before.types if t]
                def _sw2_score(sw_act):
                    slot = getattr(sw_act, "target_slot", 1) - 1
                    mon = state.p2.pokemon[slot] if 0 <= slot < len(state.p2.pokemon) else None
                    if not mon or mon.is_fainted:
                        return 999.0
                    t1 = mon.active_types[0]
                    t2 = mon.active_types[1] if len(mon.active_types) > 1 else None
                    worst_eff = max((get_type_effectiveness(ot, t1, t2) for ot in p1_types), default=1.0)
                    return worst_eff - 0.5 * mon.hp_percent
                p2_switches.sort(key=_sw2_score)
            p2_actions = (p2_moves if p2_moves else p2_actions[:1]) + p2_switches[:2]

        if not p1_actions:
            return (None, np.array([]), [], -1.0)
        if not p2_actions:
            return (p1_actions[0], np.ones(len(p1_actions)) / len(p1_actions), p1_actions, 1.0)

        n = len(p1_actions)
        m = len(p2_actions)

        active_priors = None
        if hasattr(self.evaluator, "get_policy_prior"):
            try:
                active_priors = self.evaluator.get_policy_prior(state, p1_actions)
            except Exception:
                active_priors = None

        spe1_before = p1_mon_before.effective_stat("spe") if p1_mon_before and not p1_mon_before.is_fainted else 0
        spe2_before = p2_mon_before.effective_stat("spe") if p2_mon_before and not p2_mon_before.is_fainted else 0

        # Simulate all (a1, a2) transitions and evaluate base payoffs
        transitions = []
        speed_tie_pairs = {}
        states_to_eval = []
        eval_coords = []
        base_M = np.zeros((n, m), dtype=np.float64)

        for i in range(n):
            a1 = p1_actions[i]
            for j in range(m):
                a2 = p2_actions[j]
                # Check for speed tie between damaging/status moves
                is_speed_tie = False
                if a1.action_type == ActionType.MOVE and a2.action_type == ActionType.MOVE and p1_mon_before and p2_mon_before:
                    m1_obj = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                    m2_obj = next((x for x in p2_mon_before.moves if getattr(x, "id", "") == a2.move_id), None)
                    if m1_obj and m2_obj and m1_obj.priority == m2_obj.priority and spe1_before == spe2_before:
                        is_speed_tie = True

                if is_speed_tie:
                    ns1 = simulate_turn_transition(state, a1, a2, tie_winner="p1")
                    ns2 = simulate_turn_transition(state, a1, a2, tie_winner="p2")
                    transitions.append(ns1)
                    speed_tie_pairs[(i, j)] = (ns1, ns2)
                    for ns_tie, tie_idx in [(ns1, 1), (ns2, 2)]:
                        if ns_tie.is_game_over:
                            if ns_tie.p2.is_all_fainted and not ns_tie.p1.is_all_fainted:
                                v = 15.0
                            elif ns_tie.p1.is_all_fainted and not ns_tie.p2.is_all_fainted:
                                v = -15.0
                            else:
                                states_to_eval.append(ns_tie)
                                eval_coords.append((i, j, tie_idx))
                                continue
                            base_M[i, j] += 0.5 * v
                        else:
                            states_to_eval.append(ns_tie)
                            eval_coords.append((i, j, tie_idx))
                else:
                    ns = simulate_turn_transition(state, a1, a2)
                    transitions.append(ns)
                    if ns.is_game_over:
                        if ns.p2.is_all_fainted and not ns.p1.is_all_fainted:
                            base_M[i, j] = 15.0
                        elif ns.p1.is_all_fainted and not ns.p2.is_all_fainted:
                            base_M[i, j] = -15.0
                        else:
                            states_to_eval.append(ns)
                            eval_coords.append((i, j, 0))
                    else:
                        states_to_eval.append(ns)
                        eval_coords.append((i, j, 0))

        if states_to_eval:
            if hasattr(self.evaluator, "evaluate_batch"):
                batch_vals = self.evaluator.evaluate_batch(states_to_eval)
            else:
                batch_vals = [self.evaluator.evaluate(s) for s in states_to_eval]
            for (i, j, tie_idx), v in zip(eval_coords, batch_vals):
                if tie_idx == 0:
                    base_M[i, j] = float(v)
                else:
                    base_M[i, j] += 0.5 * float(v)

        M = base_M.copy()

        # Post-transition adjustments: Material Faints, Setup Blunders, Switch Costs
        # Pre-compute whether active Pokémon hold guaranteed 1-hit KO moves or threats on their active target
        p1_has_lethal = False
        p1_has_faster_lethal = False
        if p1_mon_before and p2_mon_before and not p1_mon_before.is_fainted and not p2_mon_before.is_fainted:
            spe1 = p1_mon_before.effective_stat("spe")
            spe2 = p2_mon_before.effective_stat("spe")
            for a in p1_actions:
                if a.action_type == ActionType.MOVE:
                    mv = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a.move_id), None)
                    if mv and mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                        rolls = calculate_damage_rolls(p1_mon_before, p2_mon_before, mv, state.weather, state.terrain, attacker_side=state.p1)
                        if rolls and rolls[0] >= p2_mon_before.current_hp:
                            p1_has_lethal = True
                            if mv.priority > 0 or spe1 > spe2:
                                p1_has_faster_lethal = True
                            break

        p2_has_lethal = False
        p2_has_faster_lethal = False
        p2_has_2hko = False
        p2_has_super_effective = False
        if p1_mon_before and p2_mon_before and not p1_mon_before.is_fainted and not p2_mon_before.is_fainted:
            spe1 = p1_mon_before.effective_stat("spe")
            spe2 = p2_mon_before.effective_stat("spe")
            for a in p2_actions:
                if a.action_type == ActionType.MOVE:
                    mv = next((x for x in p2_mon_before.moves if getattr(x, "id", "") == a.move_id), None)
                    if mv and mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                        rolls = calculate_damage_rolls(p2_mon_before, p1_mon_before, mv, state.weather, state.terrain, attacker_side=state.p2)
                        if rolls:
                            if rolls[0] >= p1_mon_before.current_hp:
                                p2_has_lethal = True
                                if mv.priority > 0 or spe2 > spe1:
                                    p2_has_faster_lethal = True
                            if max(rolls) >= p1_mon_before.max_hp * 0.45:
                                p2_has_2hko = True
                        eff = get_type_effectiveness(mv.move_type, p1_mon_before.active_types[0], p1_mon_before.active_types[1] if len(p1_mon_before.active_types) > 1 else None)
                        if eff > 1.0:
                            p2_has_super_effective = True

        is_p1_debuffed = p1_mon_before and (p1_mon_before.boosts.get("spa", 0) <= -1 or p1_mon_before.boosts.get("atk", 0) <= -1)
        is_p1_threatened = p2_has_lethal or (p2_has_2hko and p2_has_super_effective)

        # Active Impotence Check: if active mon has 0 direct damage against active defender
        p1_has_zero_direct_dmg = False
        if p1_mon_before and p2_mon_before and not p1_mon_before.is_fainted and not p2_mon_before.is_fainted:
            p1_max_dd = 0
            for dm in p1_mon_before.moves:
                if dm.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL) and clean_key(dm.id) not in ("ruination", "superfang", "seismictoss", "nightshade"):
                    d_rolls = calculate_damage_rolls(p1_mon_before, p2_mon_before, dm, state.weather, state.terrain, attacker_side=state.p1)
                    if d_rolls and max(d_rolls) > p1_max_dd:
                        p1_max_dd = max(d_rolls)
            p1_has_zero_direct_dmg = (p1_max_dd == 0)

        idx = 0
        for i in range(n):
            a1 = p1_actions[i]
            m1_id = getattr(a1, "move_id", "").lower().replace("-", "").replace(" ", "") if a1.action_type == ActionType.MOVE else ""
            is_p1_setup = m1_id in ("swordsdance", "nastyplot", "calmmind", "dragondance", "quiverdance", "shellsmash", "bulkup", "shiftgear", "irondefense", "curse")
            is_p1_protect = m1_id in ("protect", "spikyshield", "detect", "banefulbunker", "silktrap")
            is_p1_recovery = m1_id in ("recover", "roost", "slackoff", "softboiled", "wish", "morningsun", "moonlight", "synthesis")

            for j in range(m):
                a2 = p2_actions[j]
                m2_id = getattr(a2, "move_id", "").lower().replace("-", "").replace(" ", "") if a2.action_type == ActionType.MOVE else ""
                is_p2_setup = m2_id in ("swordsdance", "nastyplot", "calmmind", "dragondance", "quiverdance", "shellsmash", "bulkup", "shiftgear", "irondefense", "curse")
                is_p2_protect = m2_id in ("protect", "spikyshield", "detect", "banefulbunker", "silktrap")
                is_p2_recovery = m2_id in ("recover", "roost", "slackoff", "softboiled", "wish", "morningsun", "moonlight", "synthesis")
                ns = transitions[idx]
                idx += 1

                # Derive effective defender for this column (handling opponent Tera)
                col_def_mon = p2_mon_before
                if a2.action_type == ActionType.MOVE and getattr(a2, "is_tera", False) and not state.p2.is_tera_used and p2_mon_before:
                    col_def_mon = p2_mon_before.clone()
                    col_def_mon.is_terastallized = True
                    col_def_mon.tera_type = a2.tera_type
                    apply_tera_boosts(col_def_mon)

                # Evaluate faster lethal against this specific column's defender
                col_p1_faster_lethal = False
                if p1_mon_before and col_def_mon and not p1_mon_before.is_fainted and not col_def_mon.is_fainted:
                    spe1 = p1_mon_before.effective_stat("spe")
                    spe_col = col_def_mon.effective_stat("spe")
                    for ca in p1_actions:
                        if ca.action_type == ActionType.MOVE:
                            cmv = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == ca.move_id), None)
                            if cmv and cmv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                                crolls = calculate_damage_rolls(p1_mon_before, col_def_mon, cmv, state.weather, state.terrain, attacker_side=state.p1)
                                if crolls and crolls[0] >= col_def_mon.current_hp:
                                    if cmv.priority > 0 or spe1 > spe_col:
                                        col_p1_faster_lethal = True
                                        break

                # 0. Switch-in death blunder penalty (walking directly into a lethal attack or entry hazards)
                if a1.action_type == ActionType.SWITCH:
                    target_slot1 = getattr(a1, "target_slot", 1) - 1
                    if 0 <= target_slot1 < len(state.p1.pokemon):
                        tgt = state.p1.pokemon[target_slot1]
                        if tgt.is_dead_to_hazards(state.p1.hazards):
                            safe_sw = [
                                p for s_idx, p in enumerate(state.p1.pokemon)
                                if s_idx != target_slot1 and not p.is_fainted and not p.is_dead_to_hazards(state.p1.hazards)
                            ]
                            if safe_sw:
                                M[i, j] -= 8.00
                    if 0 <= target_slot1 < len(ns.p1.pokemon):
                        entering_after = ns.p1.pokemon[target_slot1]
                        hp_lost = (tgt.current_hp - entering_after.current_hp) if tgt else 0
                        if entering_after.is_fainted:
                            M[i, j] -= 6.00
                        elif hp_lost >= entering_after.max_hp * 0.60 or entering_after.hp_percent <= 0.15:
                            M[i, j] -= 1.50
                        # Strategic free switch-in when predicting opponent Protect (if active mon lacks lethal or super-effective attack)
                        p1_has_super = any(
                            pm.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL)
                            and get_type_effectiveness(pm.move_type, col_def_mon.active_types[0], col_def_mon.active_types[1] if len(col_def_mon.active_types) > 1 else None) >= 2.0
                            for pm in p1_mon_before.moves
                        ) if p1_mon_before and col_def_mon else False

                        if is_p2_protect and not col_p1_faster_lethal and not p1_has_super and not entering_after.is_fainted and entering_after.hp_percent >= 0.40:
                            M[i, j] += 1.50

                if a2.action_type == ActionType.SWITCH:
                    target_slot2 = getattr(a2, "target_slot", 1) - 1
                    if 0 <= target_slot2 < len(ns.p2.pokemon) and ns.p2.pokemon[target_slot2].is_fainted:
                        M[i, j] += 5.00

                # 0b. Overheal & Unsafe Recovery Blunder Prevention:
                if is_p1_recovery and p1_mon_before:
                    if p1_mon_before.hp_percent >= 0.75:
                        M[i, j] -= 6.00  # Redundant overhealing when already at healthy HP
                    elif is_p1_threatened and p1_mon_before.hp_percent >= 0.55:
                        M[i, j] -= 5.00  # Healing directly into incoming lethal / super-effective hit

                # 0c. Priority Attack Preference Against Faster Lethal Threat:
                # If opponent is faster and threatens lethal damage, and we hold a priority attack (like Sucker Punch),
                # priority attack must be favored over slower damaging attacks (which would faint before striking).
                if p2_has_faster_lethal and p1_mon_before and not p1_mon_before.is_fainted and p1_sucker_streak == 0:
                    p1_has_priority_damaging = any(
                        getattr(pm, "id", "") == "suckerpunch" or getattr(pm, "priority", 0) > 0
                        for pm in p1_mon_before.moves if pm.category != MoveCategory.STATUS
                    )
                    if p1_has_priority_damaging and a1.action_type == ActionType.MOVE:
                        m1_obj_prio = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                        m1_prio = getattr(m1_obj_prio, "priority", 0) if m1_obj_prio else 0
                        if m1_id == "suckerpunch" or m1_prio > 0:
                            M[i, j] += 2.50  # Reward striking first before taking lethal damage
                        elif m1_obj_prio and m1_obj_prio.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                            M[i, j] -= 2.50  # Penalize choosing a slower damaging move over priority

                # Guaranteed Faster Lethal Dominance:
                # If we outspeed and have a guaranteed 100% OHKO move, using ANY move that fails to KO or SWITCHING is strictly dominated!
                if col_p1_faster_lethal and p1_mon_before and col_def_mon:
                    if a1.action_type == ActionType.MOVE:
                        m1_obj = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                        if m1_obj and m1_obj.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                            rolls = calculate_damage_rolls(p1_mon_before, col_def_mon, m1_obj, state.weather, state.terrain, attacker_side=state.p1)
                            if rolls and min(rolls) < col_def_mon.current_hp:
                                M[i, j] -= 5.00  # Forfeiting a guaranteed immediate KO to click a non-lethal move is strictly dominated!
                            elif rolls and min(rolls) >= col_def_mon.current_hp:
                                # When multiple moves guarantee an immediate KO, break ties in favor of higher type effectiveness / overkill
                                def_t1 = col_def_mon.active_types[0]
                                def_t2 = col_def_mon.active_types[1] if len(col_def_mon.active_types) > 1 else None
                                eff = get_type_effectiveness(m1_obj.move_type, def_t1, def_t2)
                                M[i, j] += 0.05 * eff
                    elif a1.action_type == ActionType.SWITCH:
                        M[i, j] -= 6.00  # Forfeiting a guaranteed immediate KO to switch out is strictly dominated!

                # Self-debuff move sanity & debuffed mon sacrifice elimination:
                if a1.action_type == ActionType.MOVE and p1_mon_before and col_def_mon:
                    m1_clean = clean_key(a1.move_id)
                    p1_it = clean_key(p1_mon_before.item)
                    spa_b = p1_mon_before.boosts.get("spa", 0)
                    atk_b = p1_mon_before.boosts.get("atk", 0)
                    is_choice_debuffed = p1_it in ("choicespecs", "choiceband", "choicescarf") and (
                        (spa_b <= -2 and getattr(p1_mon_before, "choice_locked_move", None) in ("dracometeor", "makeitrain", "overheat", "leafstorm", "fleurcannon")) or
                        (atk_b <= -2 and getattr(p1_mon_before, "choice_locked_move", None) in ("superpower",))
                    )
                    if is_choice_debuffed:
                        m1_obj = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                        if m1_obj:
                            rolls = calculate_damage_rolls(p1_mon_before, col_def_mon, m1_obj, state.weather, state.terrain, attacker_side=state.p1)
                            if not rolls or min(rolls) < col_def_mon.current_hp:
                                M[i, j] -= 6.00  # Continuing to attack at -2/-4/-6 SpA without KO is strictly penalized!

                    if (is_p1_debuffed or p2_has_faster_lethal) and is_p1_threatened and not col_p1_faster_lethal:
                        # Threatened mon facing faster lethal cannot kill opponent: do NOT throw it away for futile move
                        m1_obj = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                        if m1_obj:
                            rolls = calculate_damage_rolls(p1_mon_before, col_def_mon, m1_obj, state.weather, state.terrain, attacker_side=state.p1)
                            if not rolls or min(rolls) < col_def_mon.current_hp or (p2_has_faster_lethal and m1_obj.priority <= 0):
                                M[i, j] -= 4.00  # Sacrificing threatened mon when a counter-switch is available is strictly dominated!
                    elif m1_clean in ("makeitrain", "dracometeor", "overheat", "leafstorm", "superpower", "fleurcannon"):
                        m1_obj = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                        if m1_obj:
                            def_t1 = col_def_mon.active_types[0]
                            def_t2 = col_def_mon.active_types[1] if len(col_def_mon.active_types) > 1 else None
                            eff = get_type_effectiveness(m1_obj.move_type, def_t1, def_t2)
                            has_super = any(
                                om.category != MoveCategory.STATUS and get_type_effectiveness(om.move_type, def_t1, def_t2) >= 2.0
                                for om in p1_mon_before.moves
                            )
                            if eff <= 0.5 and has_super and a2.action_type == ActionType.MOVE:
                                M[i, j] -= 3.00

                    # Resisted Move Penalty when Stronger Neutral/Super-Effective Alternative Exists:
                    m1_obj = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                    if m1_obj and m1_obj.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                        def_t1 = col_def_mon.active_types[0]
                        def_t2 = col_def_mon.active_types[1] if len(col_def_mon.active_types) > 1 else None
                        m1_eff = get_type_effectiveness(m1_obj.move_type, def_t1, def_t2)
                        if m1_eff <= 0.5:
                            base_t1 = p2_mon_before.active_types[0]
                            base_t2 = p2_mon_before.active_types[1] if len(p2_mon_before.active_types) > 1 else None
                            was_base_super = get_type_effectiveness(m1_obj.move_type, base_t1, base_t2) >= 2.0
                            if not was_base_super:
                                has_better_damage = any(
                                    om.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL)
                                    and clean_key(om.id) != clean_key(m1_obj.id)
                                    and (om.base_power * get_type_effectiveness(om.move_type, def_t1, def_t2)) >= 2.0 * (m1_obj.base_power * m1_eff)
                                    for om in p1_mon_before.moves
                                )
                                if has_better_damage:
                                    M[i, j] -= 6.00

                # Defensive Tera Anticipation & All-Coverage Discipline:
                # If the opponent has NOT Terastallized yet and holds a defensive Tera type (e.g. Ting-Lu Tera Poison),
                # and our move is super effective on base (Fighting vs Dark) but resisted on Tera (Fighting vs Poison),
                # while we hold an alternative STAB/damaging move (Ground) that hits BOTH the base form neutrally (1x)
                # and hits their defensive Tera type super-effectively (2x),
                # penalize clicking the move that gets hard-walled if they Terastallize!
                if a1.action_type == ActionType.MOVE and p1_mon_before and col_def_mon and not state.p2.is_tera_used and col_def_mon.tera_type:
                    m1_obj = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                    if m1_obj and m1_obj.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                        def_base_t1 = col_def_mon.types[0]
                        def_base_t2 = col_def_mon.types[1] if len(col_def_mon.types) > 1 else None
                        base_eff = get_type_effectiveness(m1_obj.move_type, def_base_t1, def_base_t2)
                        tera_eff = get_type_effectiveness(m1_obj.move_type, col_def_mon.tera_type)
                        if base_eff >= 2.0 and tera_eff <= 0.5:
                            has_all_coverage_alt = any(
                                alt_m.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL)
                                and clean_key(alt_m.id) != clean_key(m1_obj.id)
                                and get_type_effectiveness(alt_m.move_type, col_def_mon.tera_type) >= 2.0
                                and get_type_effectiveness(alt_m.move_type, def_base_t1, def_base_t2) >= 1.0
                                for alt_m in p1_mon_before.moves
                            )
                            if has_all_coverage_alt:
                                M[i, j] -= 4.00

                # Choice Item STAB Discipline:
                # If holding Choice Specs/Band/Scarf, locking into non-STAB coverage move
                # on a speculative switch prediction is strictly penalized if active defender is not immune to STAB.
                if a1.action_type == ActionType.MOVE and p1_mon_before and col_def_mon:
                    p1_it = clean_key(p1_mon_before.item)
                    if p1_it in ("choicespecs", "choiceband", "choicescarf"):
                        m1_obj = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                        if m1_obj and m1_obj.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                            is_m1_stab = (m1_obj.move_type in p1_mon_before.types)
                            if not is_m1_stab:
                                def_t1 = col_def_mon.active_types[0]
                                def_t2 = col_def_mon.active_types[1] if len(col_def_mon.active_types) > 1 else None
                                has_effective_stab = any(
                                    sm.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL)
                                    and sm.move_type in p1_mon_before.types
                                    and get_type_effectiveness(sm.move_type, def_t1, def_t2) > 0.0
                                    for sm in p1_mon_before.moves
                                )
                                if has_effective_stab:
                                    M[i, j] -= 3.50  # Speculative non-STAB coverage lock is strictly penalized

                # Damaging Move Immunity Elimination:
                # Clicking a damaging move that deals 0 damage into an active type/ability/item immunity
                # (e.g. Ground into Flying / Levitate / Air Balloon, Normal/Fighting into Ghost, Ghost into Normal)
                # when the opponent stays in is strictly dominated.
                if a1.action_type == ActionType.MOVE and p1_mon_before and col_def_mon:
                    m1_obj = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                    if m1_obj and m1_obj.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                        rolls = calculate_damage_rolls(p1_mon_before, col_def_mon, m1_obj, state.weather, state.terrain, attacker_side=state.p1)
                        if rolls and max(rolls) == 0:
                            if a2.action_type == ActionType.MOVE:
                                M[i, j] -= 10.00  # Strictly dominated: attack into known active immunity

                # Ruination Stall Trap Penalty:
                # Ruination halves current HP and CANNOT faint a target. If target has passive recovery (Poison Heal, Leftovers, Regenerator)
                # or if the user is taking worsening Toxic damage, spamming non-lethal Ruination is an infinite stall trap.
                if m1_id == "ruination" and col_def_mon:
                    p2_ab = clean_key(col_def_mon.ability)
                    has_p_heal = (p2_ab == "poisonheal" and col_def_mon.status in (StatusCondition.TOXIC, StatusCondition.POISON)) or p2_ab == "regenerator"
                    has_lefties = clean_key(col_def_mon.item) == "leftovers"
                    if p1_mon_before and p1_mon_before.status in (StatusCondition.TOXIC, StatusCondition.POISON):
                        M[i, j] -= 6.00
                    elif has_p_heal:
                        M[i, j] -= 7.00  # Mathematical impossibility to KO a Poison Heal target with Ruination
                    elif has_lefties and col_def_mon.hp_percent <= 0.60:
                        M[i, j] -= 4.00
                    if m2_id in ("spikes", "stealthrock", "toxicspikes", "stickyweb"):
                        M[i, j] -= 6.00  # Allowing opponent to freely set hazards while clicking non-lethal Ruination is catastrophic

                # 1. Material Faint Penalty / Reward (loss of a Pokémon is a major material shift)
                if (i, j) in speed_tie_pairs:
                    ns_tie1, ns_tie2 = speed_tie_pairs[(i, j)]
                    p1_faints = 0.5 * (ns_tie1.p1.fainted_count - state.p1.fainted_count) + 0.5 * (ns_tie2.p1.fainted_count - state.p1.fainted_count)
                    p2_faints = 0.5 * (ns_tie1.p2.fainted_count - state.p2.fainted_count) + 0.5 * (ns_tie2.p2.fainted_count - state.p2.fainted_count)
                    # Speed tie lethal risk asymmetry: risking a healthy Pokémon on a 50/50 coin flip against a low HP opponent
                    p1_tie_lethal = (ns_tie2.p1.fainted_count > state.p1.fainted_count)
                    p2_tie_lethal = (ns_tie1.p2.fainted_count > state.p2.fainted_count)
                    if p1_tie_lethal and p2_tie_lethal and p1_mon_before and p2_mon_before:
                        if p1_mon_before.hp_percent > p2_mon_before.hp_percent:
                            # Asymmetric risk penalty: healthy mon risking death to kill a crippled mon
                            M[i, j] -= 6.0 * (p1_mon_before.hp_percent - p2_mon_before.hp_percent)
                else:
                    p1_faints = float(ns.p1.fainted_count - state.p1.fainted_count)
                    p2_faints = float(ns.p2.fainted_count - state.p2.fainted_count)

                if p1_faints > 0:
                    M[i, j] -= self.faint_penalty * p1_faints
                    # Severe penalty for suicide switch: sacrificing a healthy teammate on entry is catastrophic
                    if a1.action_type == ActionType.SWITCH:
                        tgt_slot = getattr(a1, "target_slot", 1) - 1
                        if 0 <= tgt_slot < len(state.p1.pokemon):
                            tgt_b = state.p1.pokemon[tgt_slot]
                            if tgt_b and tgt_b.hp_percent >= 0.40:
                                M[i, j] -= 12.00

                if p2_faints > 0:
                    M[i, j] += self.faint_penalty * p2_faints
                    if a2.action_type == ActionType.SWITCH:
                        tgt_slot2 = getattr(a2, "target_slot", 1) - 1
                        if 0 <= tgt_slot2 < len(state.p2.pokemon):
                            tgt_b2 = state.p2.pokemon[tgt_slot2]
                            if tgt_b2 and tgt_b2.hp_percent >= 0.40:
                                M[i, j] += 12.00

                # 2. Setup Satiation / Safe Setup Punish / Defensive Tera Setup
                # When defender is already in 1-shot KO range, setup moves provide 0 incremental kills.
                # When facing lethal/2HKO incoming damage that actually executes, non-Tera setup is fatal.
                # BUT when Terastallizing into an immune or resistant type, setup is safe and rewarded!
                if is_p1_setup and p1_mon_before:
                    if p1_has_lethal:
                        M[i, j] -= 10.00  # Target already in 1-shot KO range: setup provides 0 incremental value
                    a2_is_damaging = (a2.action_type == ActionType.MOVE and not (is_p2_setup or is_p2_recovery or is_p2_protect))
                    if a2_is_damaging and (p2_has_lethal or p2_has_2hko):
                        is_p1_tera = getattr(a1, "is_tera", False)
                        p1_tera_t = getattr(a1, "tera_type", None) or p1_mon_before.tera_type
                        m2_mv = next((x for x in p2_mon_before.moves if getattr(x, "id", "") == a2.move_id), None) if p2_mon_before else None
                        eff = 1.0
                        if m2_mv and is_p1_tera and p1_tera_t:
                            eff = get_type_effectiveness(m2_mv.move_type, p1_tera_t)
                        elif m2_mv and p1_mon_before:
                            eff = get_type_effectiveness(m2_mv.move_type, p1_mon_before.active_types[0], p1_mon_before.active_types[1] if len(p1_mon_before.active_types) > 1 else None)

                        if is_p1_tera and eff == 0.0:
                            M[i, j] += 2.50  # Defensive Tera grants total immunity to opponent's attack! (e.g. Flying vs Ground)
                        elif is_p1_tera and eff <= 0.5:
                            M[i, j] += 1.00  # Defensive Tera grants resistance!
                        else:
                            M[i, j] -= 4.00  # Non-Tera setup while taking 2HKO or lethal damage is a blunder!
                    elif is_p2_protect or is_p2_setup or is_p2_recovery:
                        M[i, j] += 1.50  # Golden opportunity: punish protect/status turns with free setup!

                    spa_b = p1_mon_before.boosts.get("spa", 0)
                    atk_b = p1_mon_before.boosts.get("atk", 0)
                    if (m1_id in ("nastyplot", "calmmind") and spa_b >= 2) or (m1_id in ("swordsdance", "dragondance") and atk_b >= 2):
                        M[i, j] -= 10.00  # Over-boosting penalty: already at +2 or higher, further setup is strictly wasteful!
                    # Air Balloon vulnerability: if opponent attack pops our balloon leaving us vulnerable to faster Ground lethal next turn
                    p1_end_item = ns.p1.pokemon[orig_idx1].item if 0 <= orig_idx1 < len(ns.p1.pokemon) else None
                    if p1_mon_before.item and "airballoon" in p1_mon_before.item.lower().replace("-", "").replace(" ", "") and p1_end_item is None:
                        M[i, j] -= 2.50

                if is_p2_setup and p2_mon_before:
                    if p2_has_lethal:
                        M[i, j] += 1.25
                    a1_is_damaging = (a1.action_type == ActionType.MOVE and not (is_p1_setup or is_p1_recovery or is_p1_protect))
                    if p1_has_lethal and a1_is_damaging:
                        M[i, j] += 2.00
                    elif is_p1_protect or is_p1_setup or is_p1_recovery:
                        M[i, j] -= 1.50
                    spa_b = p2_mon_before.boosts.get("spa", 0)
                    atk_b = p2_mon_before.boosts.get("atk", 0)
                    if (m2_id in ("nastyplot", "calmmind") and spa_b >= 2) or (m2_id in ("swordsdance", "dragondance") and atk_b >= 2):
                        M[i, j] += 1.0

                # 3. Insolvent Recovery Trap Mitigation
                if is_p1_recovery and p1_mon_before:
                    if p1_mon_before.current_hp >= p1_mon_before.max_hp * 0.95:
                        M[i, j] -= 25.0  # At full HP, recovery heals 0 and is completely wasted
                        if p1_has_lethal:
                            M[i, j] -= 10.0
                    elif p1_has_lethal:
                        M[i, j] -= 15.0  # When we hold a lethal attack, closing out the game takes priority
                    elif p2_has_lethal:
                        M[i, j] -= 6.00  # Do not click recovery when opponent deals lethal damage
                    else:
                        p1_hp_start = p1_mon_before.current_hp
                        p1_hp_end = ns.p1.pokemon[orig_idx1].current_hp
                        p2_hp_change = (p2_mon_before.current_hp - ns.p2.pokemon[orig_idx2].current_hp) if p2_mon_before else 0
                        if p1_hp_end <= p1_hp_start and p2_hp_change <= 0:
                            deficit_ratio = (p1_hp_start - p1_hp_end) / max(1, p1_mon_before.max_hp)
                            M[i, j] -= (0.25 + deficit_ratio)

                if is_p2_recovery and p2_mon_before:
                    if p2_mon_before.current_hp >= p2_mon_before.max_hp * 0.95:
                        M[i, j] += 3.5
                    elif p2_has_lethal:
                        M[i, j] += 2.5
                    elif p1_has_lethal:
                        M[i, j] += 2.00
                    else:
                        p2_hp_start = p2_mon_before.current_hp
                        p2_hp_end = ns.p2.pokemon[orig_idx2].current_hp
                        p1_hp_change = (p1_mon_before.current_hp - ns.p1.pokemon[orig_idx1].current_hp) if p1_mon_before else 0
                        if p2_hp_end <= p2_hp_start and p1_hp_change <= 0:
                            deficit_ratio = (p2_hp_start - p2_hp_end) / max(1, p2_mon_before.max_hp)
                            M[i, j] += (0.25 + deficit_ratio)

                # 4. Protect Decay & Stall Depletion Penalty
                if is_p1_protect and p1_mon_before:
                    if is_p2_protect:
                        M[i, j] -= 2.50  # Double Protect stall wastes turns and accomplishes nothing
                    if is_p2_setup:
                        M[i, j] -= 6.00  # Stalling against setup gives opponent free stat boosts!
                    if p2_mon_before:
                        p2_has_setup_in_set = any(
                            clean_key(m.id) in ("swordsdance", "nastyplot", "calmmind", "dragondance", "quiverdance", "shellsmash", "bulkup", "shiftgear")
                            for m in p2_mon_before.moves
                        )
                        if p2_has_setup_in_set:
                            M[i, j] -= 4.00  # Stalling into a known setup sweeper invites a free devastating dance!
                    p1_streak = getattr(p1_mon_before, "protect_streak", 0)
                    if p1_streak >= 1:
                        p1_fail_rate = 1.0 - (1.0 / (3.0 ** p1_streak))
                        if p2_has_lethal:
                            M[i, j] -= p1_fail_rate * (self.faint_penalty + 3.0)
                        M[i, j] -= 5.0 * p1_streak
                        if p1_has_lethal:
                            M[i, j] -= 4.0  # We hold a lethal attack, do not stall!
                    if p1_streak >= 2:
                        M[i, j] -= 15.0  # Hard ban on consecutive protect >= 2

                if is_p2_protect and p2_mon_before:
                    if is_p1_setup:
                        M[i, j] += 4.00  # Opponent stalling against our setup gives us free boost!
                    p2_streak = getattr(p2_mon_before, "protect_streak", 0)
                    if p2_streak >= 1:
                        p2_fail_rate = 1.0 - (1.0 / (3.0 ** p2_streak))
                        if p1_has_lethal:
                            M[i, j] += p2_fail_rate * self.faint_penalty
                        M[i, j] += 1.0 * p2_streak
                    if p2_streak >= 2:
                        M[i, j] += 3.0

                # 4b. Spiky Shield Contact Avoidance
                if is_p2_protect and p2_mon_before:
                    p2_prot_mv = getattr(p2_mon_before, "last_protect_move", "") or m2_id
                    if p2_prot_mv == "spikyshield" and a1.action_type == ActionType.MOVE:
                        mv = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                        if mv and is_contact_move(m1_id, mv.category):
                            M[i, j] -= 1.50  # Blocked by Spiky Shield + 12.5% max HP recoil damage

                # 4c. Purposeless Hazard / Phazing Penalty when opponent has no benched Pokémon
                p2_alive_bench = len(state.p2.available_switches())
                if p2_alive_bench == 0:
                    if m1_id in ("stealthrock", "spikes", "toxicspikes", "stickyweb", "whirlwind", "roar"):
                        M[i, j] -= 6.00  # Completely useless: opponent has no bench to switch in or phaze

                p1_alive_bench = len(state.p1.available_switches())
                if p1_alive_bench == 0:
                    if m2_id in ("stealthrock", "spikes", "toxicspikes", "stickyweb", "whirlwind", "roar"):
                        M[i, j] += 6.00

                # 4d. Redundant Entry Hazard Penalty (Fails on execution)
                # Setting Stealth Rock when already up, or Spikes >= 3, Toxic Spikes >= 2, Sticky Web is completely wasted and fails.
                is_m1_redundant_hazard = False
                if m1_id == "stealthrock" and Hazard.STEALTH_ROCK in state.p2.hazards:
                    is_m1_redundant_hazard = True
                elif m1_id == "spikes" and state.p2.hazards.get(Hazard.SPIKES_1, 0) >= 3:
                    is_m1_redundant_hazard = True
                elif m1_id == "toxicspikes" and state.p2.hazards.get(Hazard.TOXIC_SPIKES_1, 0) >= 2:
                    is_m1_redundant_hazard = True
                elif m1_id == "stickyweb" and Hazard.STICKY_WEB in state.p2.hazards:
                    is_m1_redundant_hazard = True

                if is_m1_redundant_hazard:
                    M[i, j] -= 15.00  # Move fails completely on Showdown!

                is_m2_redundant_hazard = False
                if m2_id == "stealthrock" and Hazard.STEALTH_ROCK in state.p1.hazards:
                    is_m2_redundant_hazard = True
                elif m2_id == "spikes" and state.p1.hazards.get(Hazard.SPIKES_1, 0) >= 3:
                    is_m2_redundant_hazard = True
                elif m2_id == "toxicspikes" and state.p1.hazards.get(Hazard.TOXIC_SPIKES_1, 0) >= 2:
                    is_m2_redundant_hazard = True
                elif m2_id == "stickyweb" and Hazard.STICKY_WEB in state.p1.hazards:
                    is_m2_redundant_hazard = True

                if is_m2_redundant_hazard:
                    M[i, j] += 15.00

                # 5. Purposeless Phazing Penalty & Suicidal Phazing Safeguard
                # Phazing moves (whirlwind, roar) operate at -6 priority, do 0 damage, and force the user to move last.
                if m1_id in ("whirlwind", "roar") and p2_mon_before:
                    p2_lethal_to_p1 = False
                    if p2_has_faster_lethal or p2_has_lethal:
                        p2_lethal_to_p1 = True
                    elif a2.action_type == ActionType.MOVE:
                        m2_mv = next((x for x in p2_mon_before.moves if getattr(x, "id", "") == a2.move_id), None)
                        if m2_mv and m2_mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                            rolls = calculate_damage_rolls(p2_mon_before, p1_mon_before, m2_mv, state.weather, state.terrain, attacker_side=state.p2)
                            if rolls and min(rolls) >= p1_mon_before.current_hp:
                                p2_lethal_to_p1 = True
                    if p2_lethal_to_p1:
                        M[i, j] -= 50.0  # Suicidal phaze: user will faint before -6 priority move executes!
                    elif m2_id in ("spikes", "stealthrock", "toxicspikes", "stickyweb"):
                        # Opponent is passively laying hazards while we phaze them away!
                        M[i, j] += 2.50
                    else:
                        p2_has_boosts = any(v > 0 for v in p2_mon_before.boosts.values())
                        p2_has_hazards = len(state.p2.hazards) > 0
                        p2_is_stall = clean_key(p2_mon_before.ability) == "poisonheal" or clean_key(p2_mon_before.item) == "leftovers"
                        if p2_is_stall and not p2_lethal_to_p1 and p2_has_hazards:
                            M[i, j] += 1.00  # Modest utility reward only when opponent hazards are active to chip bench
                        elif not p2_has_boosts and not p2_has_hazards and not p1_has_zero_direct_dmg:
                            M[i, j] -= 1.50

                if m2_id in ("whirlwind", "roar") and p1_mon_before:
                    p1_lethal_to_p2 = False
                    if p1_has_faster_lethal or p1_has_lethal:
                        p1_lethal_to_p2 = True
                    elif a1.action_type == ActionType.MOVE:
                        m1_mv = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                        if m1_mv and m1_mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                            rolls = calculate_damage_rolls(p1_mon_before, p2_mon_before, m1_mv, state.weather, state.terrain, attacker_side=state.p1)
                            if rolls and min(rolls) >= p2_mon_before.current_hp:
                                p1_lethal_to_p2 = True
                    if p1_lethal_to_p2:
                        M[i, j] += 50.0
                    elif m1_id in ("spikes", "stealthrock", "toxicspikes", "stickyweb"):
                        M[i, j] -= 3.00
                    else:
                        p1_has_boosts = any(v > 0 for v in p1_mon_before.boosts.values())
                        p1_has_hazards = len(state.p1.hazards) > 0
                        if not p1_has_boosts and not p1_has_hazards:
                            M[i, j] += 1.0

                # 5b. Purposeless Rapid Spin / Mortal Spin Penalty
                # Rapid Spin has only 50 BP. If opponent has lowered defense or is in KO range,
                # or if an effective high-power STAB alternative exists (Close Combat, Headlong Rush),
                # clicking 50 BP Rapid Spin is heavily suboptimal.
                if m1_id in ("rapidspin", "mortalspin") and p2_mon_before and p1_mon_before:
                    p1_has_hazards = len(state.p1.hazards) > 0
                    opp_def_lowered = p2_mon_before.boosts.get("def", 0) < 0
                    p1_threatened_lethal = p2_has_faster_lethal or p1_mon_before.hp_percent <= 0.40

                    has_high_power_stab = any(
                        om.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL)
                        and om.base_power >= 80
                        and om.move_type in p1_mon_before.types
                        for om in p1_mon_before.moves if clean_key(om.id) not in ("rapidspin", "mortalspin")
                    )

                    if (opp_def_lowered or p1_threatened_lethal) and has_high_power_stab:
                        M[i, j] -= 6.00

                    if not p1_has_hazards:
                        def_t1 = p2_mon_before.active_types[0]
                        def_t2 = p2_mon_before.active_types[1] if len(p2_mon_before.active_types) > 1 else None
                        has_super = any(
                            om.category != MoveCategory.STATUS and get_type_effectiveness(om.move_type, def_t1, def_t2) >= 2.0
                            for om in p1_mon_before.moves if clean_key(om.id) not in ("rapidspin", "mortalspin")
                        )
                        if has_super:
                            M[i, j] -= 2.50

                # 6. Sucker Punch Execution & Stalling Dynamics
                if m1_id == "suckerpunch":
                    # Check if Sucker Punch itself is lethal to opponent
                    sp1_mv = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == "suckerpunch"), None) if p1_mon_before else None
                    sp1_is_lethal = False
                    if sp1_mv and p2_mon_before:
                        sp1_rolls = calculate_damage_rolls(p1_mon_before, p2_mon_before, sp1_mv, state.weather, state.terrain, attacker_side=state.p1)
                        if sp1_rolls and min(sp1_rolls) >= p2_mon_before.current_hp:
                            sp1_is_lethal = True

                    opp_is_low = p2_mon_before and (p2_mon_before.current_hp <= 50 or p2_mon_before.hp_percent <= 0.25)

                    if is_p2_setup:
                        # Sucker Punch completely FAILS against setup moves, giving opponent a free boost!
                        M[i, j] -= 6.00
                    elif is_p2_protect:
                        # Sucker Punch fails 100% of the time into Protect/Spiky Shield!
                        p2_prot_streak = getattr(col_def_mon, "protect_streak", 0) if col_def_mon else 0
                        M[i, j] -= 6.00 + (3.00 * p2_prot_streak)
                    elif is_p2_recovery or a2.action_type == ActionType.SWITCH:
                        penalty = 3.50
                        if p1_has_lethal and not opp_is_low:
                            penalty += 1.50
                        M[i, j] -= penalty
                    elif a2.action_type == ActionType.MOVE and not (is_p2_setup or is_p2_recovery or is_p2_protect):
                        if sp1_is_lethal:
                            M[i, j] += 2.00  # Genuine priority lethal KO reward!
                        elif opp_is_low:
                            M[i, j] += 0.80
                        elif not sp1_is_lethal and p2_mon_before and p2_mon_before.hp_percent > 0.40:
                            # Target is healthy (>40% HP) and Sucker Punch does not KO!
                            # If we have a higher BP damaging move (e.g. Kowtow Cleave 85 BP vs Sucker Punch 70 BP),
                            # clicking non-lethal Sucker Punch is strictly penalized!
                            has_stronger_damaging = any(
                                om.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL)
                                and clean_key(om.id) != "suckerpunch"
                                and om.base_power > 70
                                for om in p1_mon_before.moves
                            )
                            if has_stronger_damaging:
                                M[i, j] -= 3.50
                        elif p1_has_lethal and not p2_has_faster_lethal:
                            # Sucker Punch does NOT KO, but we hold another move (e.g. Kowtow Cleave) that DOES KO!
                            # Clicking non-lethal Sucker Punch instead of a lethal attack is severely suboptimal.
                            M[i, j] -= 2.50

                    if p1_sucker_streak >= 1:
                        # Escalating anti-stall decay: prevent being baited into running out of PP!
                        M[i, j] -= 8.00 * p1_sucker_streak

                if m2_id == "suckerpunch":
                    sp2_mv = next((x for x in p2_mon_before.moves if getattr(x, "id", "") == "suckerpunch"), None) if p2_mon_before else None
                    sp2_is_lethal = False
                    if sp2_mv and p1_mon_before:
                        sp2_rolls = calculate_damage_rolls(p2_mon_before, p1_mon_before, sp2_mv, state.weather, state.terrain, attacker_side=state.p2)
                        if sp2_rolls and min(sp2_rolls) >= p1_mon_before.current_hp:
                            sp2_is_lethal = True

                    p1_is_low = p1_mon_before and (p1_mon_before.current_hp <= 50 or p1_mon_before.hp_percent <= 0.25)

                    if is_p1_setup:
                        if not p1_has_lethal:
                            M[i, j] += 4.50
                    elif is_p1_recovery or is_p1_protect:
                        if not (is_p1_recovery and p1_mon_before and p1_mon_before.hp_percent >= 0.95) and not (is_p1_protect and getattr(p1_mon_before, "protect_streak", 0) >= 2):
                            penalty = 2.00
                            if p2_has_lethal and not p1_is_low:
                                penalty += 1.50
                            M[i, j] += penalty
                    elif a1.action_type == ActionType.SWITCH:
                        target_slot = getattr(a1, "target_slot", 1) - 1
                        incoming_mon = state.p1.pokemon[target_slot] if 0 <= target_slot < len(state.p1.pokemon) else None
                        incoming_low = incoming_mon and (incoming_mon.current_hp <= 60 or incoming_mon.hp_percent <= 0.35)
                        all_bench_low = all(
                            p.is_fainted or p.current_hp <= 60 or p.hp_percent <= 0.35
                            for idx_p, p in enumerate(state.p1.pokemon) if idx_p != orig_idx1
                        )
                        if last_action_was_switch or incoming_low or all_bench_low:
                            # Chaining switches or switching to a crippled bench into Sucker Punch is suicidal!
                            # Active Pokémon must commit an attack rather than throwing away teammates.
                            M[i, j] -= 5.00
                        else:
                            penalty = 2.00
                            if p2_has_lethal and not p1_is_low:
                                penalty += 1.50
                            M[i, j] += penalty
                    elif a1.action_type == ActionType.MOVE and not (is_p1_setup or is_p1_recovery or is_p1_protect):
                        if sp2_is_lethal:
                            M[i, j] -= 2.00
                        elif p1_is_low:
                            M[i, j] -= 0.80
                        elif p2_has_lethal and not p1_has_faster_lethal:
                            M[i, j] += 2.50

                # 7. Terastallization Defensive Weakness & Option Economics Penalty
                # Terastallizing into a type that is weak to the opponent's STAB types
                # or burning once-per-match Tera on an opponent already near death is penalized.
                if getattr(a1, "is_tera", False) and not state.p1.is_tera_used:
                    tera_t = getattr(a1, "tera_type", None) or (p1_mon_before.tera_type if p1_mon_before else None)
                    if tera_t and p2_mon_before:
                        if p2_mon_before.current_hp <= 50 or p2_mon_before.hp_percent <= 0.25:
                            M[i, j] -= 5.00  # Squandering once-per-match Tera on a crippled target
                        for ot in p2_mon_before.types:
                            if ot and get_type_effectiveness(ot, tera_t) > 1.0:
                                M[i, j] -= 1.50
                                break

                    if p1_mon_before:
                        spa_boost = p1_mon_before.boosts.get("spa", 0)
                        atk_boost = p1_mon_before.boosts.get("atk", 0)
                        if spa_boost <= -2 or atk_boost <= -2:
                            M[i, j] -= 5.00

                        if p1_mon_before.hp_percent <= 0.40 and p2_has_faster_lethal and tera_t:
                            provides_resist = any(
                                get_type_effectiveness(ot, tera_t) < 1.0
                                for ot in (p2_mon_before.types if p2_mon_before else []) if ot
                            )
                            if not provides_resist:
                                M[i, j] -= 6.00

                    # Early-Game Tera Preservation:
                    # Terastallization is an irreplaceable once-per-battle team resource.
                    # Do not burn Tera on Turn 1-4 unless it scores an immediate KO or saves from lethal damage.
                    p2_end_hp = ns.p2.pokemon[orig_idx2].current_hp if 0 <= orig_idx2 < len(ns.p2.pokemon) else 0
                    p1_end_hp = ns.p1.pokemon[orig_idx1].current_hp if 0 <= orig_idx1 < len(ns.p1.pokemon) else 0
                    is_tera_ko = (p2_end_hp == 0)
                    is_tera_save = (p1_end_hp > 0 and p2_has_lethal)
                    if not is_tera_ko and not is_tera_save:
                        p1_alive = len(state.p1.pokemon) - state.p1.fainted_count
                        if state.turn <= 4 or p1_alive >= 5:
                            M[i, j] -= 4.00  # Strong early-game preservation penalty
                        else:
                            M[i, j] -= 1.00  # Mild mid-game preservation penalty

                    # Redundant / Wasteful Tera Squandering Guard:
                    # If the base attack without Terastallization already KOs the target,
                    # burning the team's once-per-battle Tera provides zero marginal value and is heavily penalized!
                    if is_tera_ko and not is_tera_save and a1.action_type == ActionType.MOVE and p1_mon_before and p2_mon_before:
                        m1_mv = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                        if m1_mv and m1_mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                            base_rolls = calculate_damage_rolls(
                                p1_mon_before, p2_mon_before, m1_mv,
                                state.weather, state.terrain, attacker_side=state.p1
                            )
                            if base_rolls and min(base_rolls) >= p2_mon_before.current_hp:
                                # Non-Tera version already secures guaranteed KO!
                                M[i, j] -= 6.00

                if getattr(a2, "is_tera", False) and not state.p2.is_tera_used:
                    tera_t = getattr(a2, "tera_type", None) or (p2_mon_before.tera_type if p2_mon_before else None)
                    if tera_t and p1_mon_before:
                        for ot in p1_mon_before.types:
                            if ot and get_type_effectiveness(ot, tera_t) > 1.0:
                                M[i, j] += 1.50
                                break

        # 8. Offensive Initiative & Chip Damage Tie-Breaker
        # Ensures that even under impending lethal threats, damaging attacks are strictly preferred
        # over futile status/setup moves by providing proportional reward for opponent HP reduction.
        for i, a1 in enumerate(p1_actions):
            if a1.action_type == ActionType.MOVE and p1_mon_before and p2_mon_before:
                mv = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                if mv and mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                    for j in range(m):
                        ns = transitions[i * m + j]
                        p2_end_hp = ns.p2.pokemon[orig_idx2].current_hp if 0 <= orig_idx2 < len(ns.p2.pokemon) else 0
                        dmg_dealt = max(0, p2_mon_before.current_hp - p2_end_hp)
                        if dmg_dealt > 0:
                            dmg_ratio = dmg_dealt / max(1, p2_mon_before.max_hp)
                            M[i, j] += 0.05 * dmg_ratio
                            if p2_end_hp == 0:
                                M[i, j] += 0.50  # Knockout execution reward!

        # Apply tempo cost to voluntary switches (game-theoretic penalty for losing offensive initiative)
        opp_in_exec_range = p2_mon_before and not p2_mon_before.is_fainted and (p2_mon_before.current_hp <= 50 or p2_mon_before.hp_percent <= 0.25)
        p1_in_exec_range = p1_mon_before and not p1_mon_before.is_fainted and (p1_mon_before.current_hp <= 50 or p1_mon_before.hp_percent <= 0.25)

        for i, a1 in enumerate(p1_actions):
            if a1.action_type == ActionType.SWITCH:
                target_slot1 = getattr(a1, "target_slot", 1) - 1
                tgt = state.p1.pokemon[target_slot1] if 0 <= target_slot1 < len(state.p1.pokemon) else None

                is_defensive_counter = False
                p1_fallen = sum(1 for p in state.p1.pokemon if p.is_fainted)
                is_early_kingambit = tgt and (clean_key(tgt.species) == "kingambit" or clean_key(tgt.ability) == "supremeoverlord") and p1_fallen < 3

                p1_it = clean_key(p1_mon_before.item) if p1_mon_before else ""
                spa_b = p1_mon_before.boosts.get("spa", 0) if p1_mon_before else 0
                atk_b = p1_mon_before.boosts.get("atk", 0) if p1_mon_before else 0
                is_choice_debuffed = p1_it in ("choicespecs", "choiceband", "choicescarf") and (
                    (spa_b <= -2 and getattr(p1_mon_before, "choice_locked_move", None) in ("dracometeor", "makeitrain", "overheat", "leafstorm", "fleurcannon")) or
                    (atk_b <= -2 and getattr(p1_mon_before, "choice_locked_move", None) in ("superpower",))
                )

                if not is_early_kingambit and not p1_has_faster_lethal and tgt and not tgt.is_fainted and tgt.hp_percent >= 0.40 and not tgt.is_dead_to_hazards(state.p1.hazards):
                    if is_choice_debuffed:
                        is_defensive_counter = True
                    elif ((is_p1_debuffed or p2_has_faster_lethal) and is_p1_threatened) or p1_has_zero_direct_dmg:
                        if p2_mon_before:
                            p2_stabs = [t for t in p2_mon_before.types if t]
                            tgt_t1 = tgt.active_types[0]
                            tgt_t2 = tgt.active_types[1] if len(tgt.active_types) > 1 else None
                            tgt_item_clean = clean_key(tgt.item)
                            tgt_ab_clean = clean_key(tgt.ability)
                            effs = []
                            for t in p2_stabs:
                                eff = get_type_effectiveness(t, tgt_t1, tgt_t2)
                                # Ground immunity from Air Balloon or Levitate
                                if t == PokemonType.GROUND and ("airballoon" in tgt_item_clean or tgt_ab_clean in ("levitate", "eartheater")):
                                    eff = 0.0
                                elif t == PokemonType.WATER and tgt_ab_clean in ("waterabsorb", "stormdrain", "dryskin"):
                                    eff = 0.0
                                elif t == PokemonType.FIRE and tgt_ab_clean == "flashfire":
                                    eff = 0.0
                                elif t == PokemonType.ELECTRIC and tgt_ab_clean in ("voltabsorb", "lightningrod", "motordrive"):
                                    eff = 0.0
                                effs.append(eff)

                            # A true defensive counter must NOT be weak to any of the opponent's primary STAB attacks!
                            if not any(e > 1.0 for e in effs):
                                if any(e == 0.0 for e in effs):
                                    is_defensive_counter = True
                                    M[i, :] += 0.50  # Extra reward for full type/item immunity!
                                elif any(e <= 0.5 for e in effs):
                                    is_defensive_counter = True

                if is_defensive_counter:
                    pen = 0.0  # Zero switch penalty for a justified defensive pivot
                    M[i, :] += 2.00  # Defensive repositioning reward!
                else:
                    pen = self.switch_penalty + (2.00 if opp_in_exec_range else 0.0)
                    if p1_has_faster_lethal:
                        pen += 5.00  # Strictly dominated: forfeits guaranteed free kill to take unpunished damage

                # Prevent back-to-back ping-pong switching (applies universally)
                if last_action_was_switch:
                    pen += 8.00  # Severe penalty for chained switching (prevents A -> B -> A -> B loop)

                # Active Board Dominance & Immunity Preservation:
                # If the active mon is already immune or completely walls the opponent's attacks (e.g. Air Balloon Gholdengo vs Gliscor),
                # voluntary switching OUT of an immune/dominant matchup is strictly dominated!
                if p1_mon_before and p2_mon_before and not p1_mon_before.is_fainted and not p2_has_faster_lethal:
                    p2_stabs = [t for t in p2_mon_before.types if t]
                    is_active_immune = any(
                        get_type_effectiveness(ot, p1_mon_before.active_types[0], p1_mon_before.active_types[1] if len(p1_mon_before.active_types) > 1 else None) == 0.0
                        or (ot == PokemonType.GROUND and ("airballoon" in clean_key(p1_mon_before.item) or clean_key(p1_mon_before.ability) in ("levitate", "eartheater")))
                        for ot in p2_stabs
                    )
                    if is_active_immune:
                        pen += 8.00  # Strictly dominated: forfeits total board control / active immunity to switch out

                # Never throw away active positive stat boosts on voluntary switch
                if p1_mon_before:
                    boost_sum = sum(v for v in p1_mon_before.boosts.values() if v > 0)
                    if boost_sum > 0:
                        pen += 1.50 * boost_sum
                    # Never throw away single-use Booster Energy Attack/Speed boost
                    if getattr(p1_mon_before, "booster_stat", None):
                        pen += 5.00

                # Kingambit Supreme Overlord Early Switch Preservation:
                # Kingambit is the team's late-game cleaner whose power scales +10% per fainted ally (Supreme Overlord).
                # Switching Kingambit in early (when >= 3 teammates are still alive) wastes its sweeping potential,
                # exposes it to chip and hazards, and is heavily penalized unless active mon is fainted.
                if is_early_kingambit and p1_mon_before and not p1_mon_before.is_fainted:
                    pen += 8.00 * (3 - p1_fallen)

                M[i, :] -= pen
        for j, a2 in enumerate(p2_actions):
            if a2.action_type == ActionType.SWITCH:
                pen = self.switch_penalty + (2.00 if p1_in_exec_range else 0.0)
                if p2_has_faster_lethal:
                    pen += 5.00
                if p2_mon_before:
                    boost_sum = sum(v for v in p2_mon_before.boosts.values() if v > 0)
                    if boost_sum > 0:
                        pen += 1.50 * boost_sum
                    if getattr(p2_mon_before, "booster_stat", None):
                        pen += 3.00
                M[:, j] += pen

        # Depth > 1: Candidate Beam Search Lookahead
        # Select viable candidate actions from Depth-1 game and deeply evaluate candidate subgames
        cand_p1 = None
        cand_p2 = None
        if depth > 1:
            pi1_imm, pi2_imm, _ = solve_zero_sum_game(M, p1_prior=active_priors)
            if cand_beam is not None:
                max_cand_p1, max_cand_p2 = cand_beam
            elif depth >= 3:
                max_cand_p1 = 3
                max_cand_p2 = 3
            else:
                max_cand_p1 = 4
                max_cand_p2 = 6

            # Player 1 Candidates: Nash support + top expected payoffs
            cand_p1 = [i for i, p in enumerate(pi1_imm) if p > 0.01]
            if len(cand_p1) < max_cand_p1:
                ev1 = M @ pi2_imm
                for idx_p1 in np.argsort(-ev1):
                    if idx_p1 not in cand_p1:
                        cand_p1.append(idx_p1)
                    if len(cand_p1) >= min(max_cand_p1, n):
                        break

            # Player 2 Candidates: Nash support + best responses against P1
            cand_p2 = [j for j, p in enumerate(pi2_imm) if p > 0.01]
            if len(cand_p2) < max_cand_p2:
                ev2 = pi1_imm @ M
                for idx_p2 in np.argsort(ev2):
                    if idx_p2 not in cand_p2:
                        cand_p2.append(idx_p2)
                    if len(cand_p2) >= min(max_cand_p2, m):
                        break

            child_beam = (2, 2) if (depth - 1) >= 2 else None
            is_child_leaf = (depth - 1 == 1)

            for i in cand_p1:
                a1 = p1_actions[i]
                for j in cand_p2:
                    a2 = p2_actions[j]
                    if (i, j) in speed_tie_pairs:
                        ns1, ns2 = speed_tie_pairs[(i, j)]
                        if not ns1.is_game_over:
                            _, _, _, v1 = self.resolve_turn(
                                ns1, depth=depth - 1, sample=False,
                                last_action_was_switch=(a1.action_type == ActionType.SWITCH),
                                cand_beam=child_beam,
                                is_leaf_eval=is_child_leaf
                            )
                            v1 = v1 * 0.90
                        else:
                            v1 = 15.0 if (ns1.p2.is_all_fainted and not ns1.p1.is_all_fainted) else (-15.0 if (ns1.p1.is_all_fainted and not ns1.p2.is_all_fainted) else self.evaluator.evaluate(ns1))

                        if not ns2.is_game_over:
                            _, _, _, v2 = self.resolve_turn(
                                ns2, depth=depth - 1, sample=False,
                                last_action_was_switch=(a1.action_type == ActionType.SWITCH),
                                cand_beam=child_beam,
                                is_leaf_eval=is_child_leaf
                            )
                            v2 = v2 * 0.90
                        else:
                            v2 = 15.0 if (ns2.p2.is_all_fainted and not ns2.p1.is_all_fainted) else (-15.0 if (ns2.p1.is_all_fainted and not ns2.p2.is_all_fainted) else self.evaluator.evaluate(ns2))
                        deep_val = 0.5 * v1 + 0.5 * v2
                    else:
                        ns = transitions[i * m + j]
                        if not ns.is_game_over:
                            _, _, _, deep_val = self.resolve_turn(
                                ns, depth=depth - 1, sample=False,
                                p1_sucker_streak=0,
                                last_action_was_switch=(a1.action_type == ActionType.SWITCH),
                                cand_beam=child_beam,
                                is_leaf_eval=is_child_leaf
                            )
                            deep_val = deep_val * 0.90
                        else:
                            deep_val = 15.0 if (ns.p2.is_all_fainted and not ns.p1.is_all_fainted) else (-15.0 if (ns.p1.is_all_fainted and not ns.p2.is_all_fainted) else self.evaluator.evaluate(ns))

                    diff = deep_val - base_M[i, j]
                    M[i, j] += diff

        # Solve for Nash Equilibrium (Minimax Mixed Strategy on extensive-form payoffs M)
        if depth > 1 and cand_p1 and cand_p2:
            sub_M = M[np.ix_(cand_p1, cand_p2)]
            sub_prior = active_priors[cand_p1] if (active_priors is not None and len(active_priors) == n) else None
            p1_sub_strat, p2_sub_strat, game_value = solve_zero_sum_game(sub_M, p1_prior=sub_prior)
            p1_strat = np.zeros(n, dtype=np.float64)
            for idx_c, p in zip(cand_p1, p1_sub_strat):
                p1_strat[idx_c] = p
            p2_strat = np.zeros(m, dtype=np.float64)
            for idx_c, p in zip(cand_p2, p2_sub_strat):
                p2_strat[idx_c] = p
        else:
            p1_strat, p2_strat, game_value = solve_zero_sum_game(M, p1_prior=active_priors)

        # Select action
        if sample:
            chosen_idx = np.random.choice(n, p=p1_strat)
        else:
            chosen_idx = int(np.argmax(p1_strat))

        return p1_actions[chosen_idx], p1_strat, p1_actions, game_value
