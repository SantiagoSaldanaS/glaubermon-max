"""Depth-Limited Simultaneous-Move Subgame Search Engine."""

import random
from typing import List, Optional, Tuple
import numpy as np
from glaubermon.core.battle_state import BattleState
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.actions import Action, MoveAction, SwitchAction
from glaubermon.core.types import ActionType, Hazard, PokemonType, MoveCategory, StatusCondition, Terrain, Weather
from glaubermon.core.constants import get_type_effectiveness, clean_key
from glaubermon.inference.damage_calc import calculate_damage_rolls, is_contact_move, is_removable_item
from glaubermon.search.evaluators import StateEvaluator, HeuristicEvaluator
from glaubermon.search.matrix_solver import solve_zero_sum_game
from glaubermon.core.field_mechanics import (effective_weather,effective_speed,move_priority,accuracy_chance,
    resolved_move_type,consume_lum_berry,set_weather,set_terrain,sync_paradox,weather_residual,hit_count,WEATHER_MOVES,WEATHER_ABILITIES,TERRAIN_MOVES,TERRAIN_ABILITIES)


def speed_order_key(mon, side, state):
    """Comparable speed within one priority bracket, including public field effects."""
    weather = effective_weather(state.weather,state.p1.active_pokemon,state.p2.active_pokemon)
    speed = effective_speed(mon,side,weather,state.terrain)
    return -speed if state.trick_room else speed


def reset_on_switch(mon):
    """Switch-out clears temporary state; major status persists except Natural Cure."""
    mon.boosts = {key:0 for key in mon.boosts}
    mon.choice_locked_move = None
    mon.last_move = None
    mon.protect_streak = 0
    mon.protect_success_rate = 0.0
    mon.booster_stat = None
    mon.type_override = None
    mon.toxic_counter = 0
    mon.volatiles.clear()
    if clean_key(mon.ability) == "naturalcure":
        mon.status = StatusCondition.NONE
        mon.status_turns = 0
    if not mon.is_fainted and clean_key(mon.ability) == "regenerator":
        mon.heal(max(1, mon.max_hp // 3))


def apply_entry_hazards(state: BattleState, side_idx: int, mon_idx: int):
    """Apply hazard damage (Stealth Rock, Spikes) to entering Pokémon."""
    side = state.p1 if side_idx == 1 else state.p2
    if 0 <= mon_idx < len(side.pokemon):
        mon = side.pokemon[mon_idx]
        dmg = mon.calculate_hazard_damage(side.hazards)
        if dmg > 0:
            mon.take_damage(dmg)
        if mon.is_fainted or clean_key(mon.item) == "heavydutyboots" or not mon.is_grounded():
            return
        toxic_layers = side.hazards.get(Hazard.TOXIC_SPIKES_1,0)
        if toxic_layers:
            if PokemonType.POISON in mon.active_types:
                side.hazards.pop(Hazard.TOXIC_SPIKES_1,None)
            elif (PokemonType.STEEL not in mon.active_types and mon.status == StatusCondition.NONE
                  and clean_key(mon.ability) not in ("immunity","pastelveil") and state.terrain != Terrain.MISTY):
                mon.status = StatusCondition.TOXIC if toxic_layers >= 2 else StatusCondition.POISON
                mon.toxic_counter = 0
        if side.hazards.get(Hazard.STICKY_WEB) and clean_key(mon.ability) not in ("clearbody","whitesmoke","fullmetalbody") and clean_key(mon.item) != "clearamulet":
            delta = 1 if clean_key(mon.ability)=="contrary" else (-2 if clean_key(mon.ability)=="simple" else -1)
            mon.boosts["spe"] = max(-6,min(6,mon.boosts.get("spe",0)+delta))


def apply_entry_abilities(state,side_idx):
    side,other = (state.p1,state.p2) if side_idx == 1 else (state.p2,state.p1)
    mon,foe = side.active_pokemon,other.active_pokemon
    if not mon or mon.is_fainted:
        return
    ability = clean_key(mon.ability)
    if ability.startswith("embodyaspect") and mon.is_terastallized:
        stat={"embodyaspectwellspring":"spd","embodyaspecthearthflame":"atk","embodyaspectcornerstone":"def"}.get(ability,"spe")
        mon.boosts[stat] = min(6,mon.boosts.get(stat,0)+1)
    if ability in WEATHER_ABILITIES:set_weather(state,WEATHER_ABILITIES[ability],mon)
    if ability in TERRAIN_ABILITIES:set_terrain(state,TERRAIN_ABILITIES[ability],mon)
    sync_paradox(state)
    if ability in ("dauntlessshield","intrepidsword"):
        flag = "shield_boosted" if ability == "dauntlessshield" else "sword_boosted"
        if not getattr(mon,flag,False):
            key = "def" if ability == "dauntlessshield" else "atk"
            mon.boosts[key] = min(6,mon.boosts.get(key,0)+1)
            setattr(mon,flag,True)
    if ability == "intimidate" and foe and not foe.is_fainted and "substitute" not in foe.volatiles:
        foe_ability = clean_key(foe.ability)
        if foe_ability in ("clearbody","whitesmoke","fullmetalbody","innerfocus","oblivious","owntempo","scrappy") or clean_key(foe.item)=="clearamulet":
            return
        target = mon if foe_ability == "mirrorarmor" else foe
        delta = 1 if foe_ability in ("contrary","guarddog") else (-2 if foe_ability=="simple" else -1)
        old = target.boosts.get("atk",0)
        target.boosts["atk"] = max(-6,min(6,old+delta))
        if target is foe and delta < 0 and old > -6:
            if foe_ability == "defiant": foe.boosts["atk"] = min(6,foe.boosts["atk"]+2)
            if foe_ability == "competitive": foe.boosts["spa"] = min(6,foe.boosts.get("spa",0)+2)
        if foe_ability == "rattled": foe.boosts["spe"] = min(6,foe.boosts.get("spe",0)+1)


def apply_tera_boosts(mon: Optional[Pokemon]):
    """Apply Embody Aspect stat boosts when Ogerpon Terastallizes in Gen 9."""
    if not mon:
        return
    spec_clean = clean_key(mon.species)
    item_clean = clean_key(mon.item)
    if "ogerpon" in spec_clean:
        if "hearthflame" in spec_clean or "hearthflame" in item_clean:
            mon.ability = "embodyaspecthearthflame"
            mon.boosts["atk"] = min(6, mon.boosts.get("atk", 0) + 1)
        elif "cornerstone" in spec_clean or "cornerstone" in item_clean:
            mon.ability = "embodyaspectcornerstone"
            mon.boosts["def"] = min(6, mon.boosts.get("def", 0) + 1)
        elif "wellspring" in spec_clean or "wellspring" in item_clean:
            mon.ability = "embodyaspectwellspring"
            mon.boosts["spd"] = min(6, mon.boosts.get("spd", 0) + 1)
        else:
            mon.ability = "embodyaspectteal"
            # Teal Mask (standard) grants +1 Speed
            mon.boosts["spe"] = min(6, mon.boosts.get("spe", 0) + 1)


def perform_switch(state, side_idx, target_slot):
    side = state.p1 if side_idx == 1 else state.p2
    if target_slot not in side.available_switches():
        raise ValueError("Invalid replacement slot")
    outgoing_index = side.active_index
    if side.active_pokemon:
        reset_on_switch(side.active_pokemon)
    for other_side in (state.p1,state.p2):
        for mon in other_side.pokemon:
            for key in ("partiallytrapped","trapped"):
                if mon.volatiles.get(key,{}).get("source") == (side_idx,outgoing_index):
                    mon.volatiles.pop(key,None)
    side.active_index = target_slot
    side.active_pokemon.protean_used = False
    apply_entry_hazards(state, side_idx, target_slot)
    consume_lum_berry(side.active_pokemon)
    apply_entry_abilities(state, side_idx)


def finalized_state(state):
    # Showdown clears volatile conditions when a faint is processed.
    for side in (state.p1,state.p2):
        for mon in side.pokemon:
            if mon.is_fainted:
                mon.volatiles.clear()
                mon.boosts = {stat:0 for stat in mon.boosts}
                mon.booster_stat = None
                mon.choice_locked_move = None
                mon.last_move = None
                mon.type_override = None
                mon.is_terastallized = False
    return state


def simulate_turn_transition(
    state: BattleState,
    a1: Action,
    a2: Action,
    tie_winner: Optional[str] = None,
    *,
    sample_outcomes: bool = False,
    rng: Optional[random.Random] = None,
) -> BattleState:
    """Advance the internal simulator, not a complete Showdown implementation.

    Search retains deterministic damage/Protect approximations by default.
    sample_outcomes draws speed ties, accuracy, critical hits, damage and Protect
    for rollout experiments. Pass a dedicated seeded Random for reproducibility.
    Sampled transitions leave fainted actives in place for the caller to replace.
    """
    rng = rng if rng is not None else random
    s = state.clone()
    sync_paradox(s)
    for side in (s.p1,s.p2):
        for mon in side.pokemon:
            for move in mon.moves:move.request_disabled = False
    if not s.pending_switches:
        s.pending_switches = tuple(i for i,side in ((1,s.p1),(2,s.p2))
            if side.active_pokemon and side.active_pokemon.is_fainted and not side.is_all_fainted)
    p1_active = s.p1.active_pokemon
    p2_active = s.p2.active_pokemon
    if p1_active and not s.pending_switches:
        p1_active.is_protected = False
        p1_active.protect_success_rate = 0.0
    if p2_active and not s.pending_switches:
        p2_active.is_protected = False
        p2_active.protect_success_rate = 0.0

    resuming = bool(s.pending_switches)
    saved = s.continuation
    if resuming:
        for side_idx, action in ((1, a1), (2, a2)):
            if side_idx in s.pending_switches:
                side = s.p1 if side_idx == 1 else s.p2
                if action is None or action.action_type != ActionType.SWITCH or action.target_slot-1 not in side.available_switches():
                    raise ValueError("A pending replacement requires a legal switch")
            elif action is not None:
                raise ValueError("The other player must wait during replacement")
    p1_switched = p2_switched = False
    switch_actions = [(i,a) for i,a in ((1,a1),(2,a2)) if a is not None and a.action_type == ActionType.SWITCH]
    # Queue speed belongs to the outgoing Pokemon, before either entry callback.
    switch_actions.sort(key=lambda pair:speed_order_key((s.p1 if pair[0]==1 else s.p2).active_pokemon,s.p1 if pair[0]==1 else s.p2,s),reverse=True)
    if len(switch_actions)==2 and sample_outcomes:
        first,second=(s.p1 if switch_actions[0][0]==1 else s.p2),(s.p1 if switch_actions[1][0]==1 else s.p2)
        if speed_order_key(first.active_pokemon,first,s)==speed_order_key(second.active_pokemon,second,s) and rng.random()<.5:switch_actions.reverse()
    for side_idx, action in switch_actions:
        if action is not None and action.action_type == ActionType.SWITCH:
            if not resuming and s.is_trapped(side_idx):
                raise ValueError("A trapped Pokémon cannot switch voluntarily")
            perform_switch(s, side_idx, action.target_slot-1)
            if side_idx == 1: p1_switched = True
            else: p2_switched = True
    p1_active, p2_active = s.p1.active_pokemon, s.p2.active_pokemon
    if resuming:
        # Hazard KOs require another decision without advancing the clock.
        s.pending_switches = tuple(i for i in s.pending_switches
            if (s.p1 if i == 1 else s.p2).active_pokemon.is_fainted
            and not (s.p1 if i == 1 else s.p2).is_all_fainted)
        if s.pending_switches or s.is_game_over:
            return finalized_state(s)
        if saved is None:
            s.pending_switches = tuple(i for i,side in ((1,s.p1),(2,s.p2))
                if side.active_pokemon and side.active_pokemon.is_fainted and not side.is_all_fainted)
            if s.pending_switches:
                return finalized_state(s)
            s.turn += 1
            return finalized_state(s)  # End-of-turn replacement, no second residual tick or attack.
        s.continuation = None

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
    def selected_move(action, mon):
        if action is None or action.action_type != ActionType.MOVE or not mon or mon.is_fainted:
            return None
        if action.move_id == "struggle":
            return Move(id="struggle", name="Struggle", move_type=PokemonType.UNKNOWN,
                        category=MoveCategory.PHYSICAL, base_power=50,
                        always_hits=True, is_contact=True)
        slot = action.move_slot - 1
        if 0 <= slot < len(mon.moves) and mon.moves[slot].pp > 0:
            return mon.moves[slot]
        return None

    m1 = selected_move(a1, p1_active)
    m2 = selected_move(a2, p2_active)

    # Determine move order (Priority first, then Effective Speed)
    if m1 and m2:
        prio1 = move_priority(p1_active,m1)
        prio2 = move_priority(p2_active,m2)
        spe1 = speed_order_key(p1_active, s.p1, s)
        spe2 = speed_order_key(p2_active, s.p2, s)

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
            winner = tie_winner
            if winner is None and sample_outcomes:
                winner = rng.choice(("p1", "p2"))
            if winner == "p2":
                order = [("p2", m2), ("p1", m1)]
            else:
                order = [("p1", m1), ("p2", m2)]
    elif m1:
        order = [("p1", m1)]
    elif m2:
        order = [("p2", m2)]
    else:
        order = []

    # A pivot pauses this same turn. Preserve the original user's queued move,
    # PP, Protect and flinch state; the incoming Pokémon cannot inherit an action.
    if resuming:
        moved_players = set(saved["moved"])
        flinched = set(saved["flinched"])
        selected_users = {who:(s.p1 if who == "p1" else s.p2).pokemon[idx]
                          for who,idx in saved["users"].items()}
        order = [(who, selected_users[who].moves[slot] if slot >= 0 else move)
                 for who,slot,move in saved["order"]]
        m1, m2 = saved["m1"], saved["m2"]
    else:
        moved_players = set()
        flinched = set()
        selected_users = {"p1":p1_active,"p2":p2_active}
    for order_index, (player, move) in enumerate(order):
        attacker = p1_active if player == "p1" else p2_active
        defender = p2_active if player == "p1" else p1_active

        if attacker is not selected_users[player]:
            continue  # A Pokémon dragged out before its turn cannot pass its move to the replacement.
        if s.is_game_over:break
        if attacker and not attacker.is_fainted and defender:
            execution_priority = move_priority(attacker,move)
            if move.id != "struggle":
                forced = attacker.volatiles.get("encore",{}).get("move")
                if forced:
                    move = next((m for m in attacker.moves if m.id == forced),move)
            m_id = move.id.lower().replace(" ", "").replace("-", "")
            selected = move
            is_protection = move.is_protect or m_id in ("protect", "spikyshield", "detect", "banefulbunker", "silktrap")
            if not is_protection:
                attacker.protect_streak = 0

            # Major status is checked before PP is consumed. Unknown sleep duration
            # is still an explicit approximation in search; rollouts use Showdown.
            if attacker.status == StatusCondition.SLEEP:
                attacker.status_turns -= 1 + int(clean_key(attacker.ability) == "earlybird")
                if attacker.status_turns <= 0:
                    attacker.status = StatusCondition.NONE
                    attacker.status_turns = 0
                elif not move.sleep_usable:
                    moved_players.add(player)
                    continue
            if attacker.status == StatusCondition.FREEZE:
                defrost = move.defrost
                if defrost or (sample_outcomes and rng.random() < 0.20):
                    attacker.status = StatusCondition.NONE
                else:
                    moved_players.add(player)
                    continue
            if player in flinched:
                moved_players.add(player)
                continue
            if attacker.volatiles.get("disable",{}).get("move") == m_id:
                moved_players.add(player)
                continue
            if "taunt" in attacker.volatiles and move.category == MoveCategory.STATUS and m_id != "mefirst":
                moved_players.add(player)
                continue
            confusion = attacker.volatiles.get("confusion")
            if confusion:
                if confusion.get("time", -1) > 0:
                    confusion["time"] -= 1
                if confusion.get("time") == 0:
                    attacker.volatiles.pop("confusion")
                elif sample_outcomes and rng.random() < .33:
                    level_term = 2 * attacker.level // 5 + 2
                    def confusion_stat(key):
                        stage = attacker.boosts.get(key,0)
                        base = attacker.raw_stats.get(key,100)
                        return base * (2+stage) // 2 if stage >= 0 else base * 2 // (2-stage)
                    base = ((level_term * 40 * confusion_stat("atk") // max(1,confusion_stat("def"))) // 50 + 2) % 65536
                    attacker.take_damage(max(1,base * rng.randint(85,100) // 100))
                    moved_players.add(player)
                    continue
            if attacker.status == StatusCondition.PARALYSIS and sample_outcomes and rng.random() < 0.25:
                moved_players.add(player)
                continue
            attacker.last_move = move.id
            atk_it = clean_key(attacker.item)
            if atk_it in ("choicespecs", "choiceband", "choicescarf") and not attacker.choice_locked_move:
                attacker.choice_locked_move = m_id
            sleep_talk_failed = False
            if m_id == "sleeptalk":
                if attacker.status != StatusCondition.SLEEP:
                    sleep_talk_failed = True
                else:
                    options = [m for m in attacker.moves if m.sleep_talk_callable]
                    if options:
                        move = rng.choice(options)
                        m_id = move.id
                    else:
                        sleep_talk_failed = True

            # Misses and failed moves consume PP; being KO'd before acting does not.
            # Deduct the selected move, not a move called by Sleep Talk.
            if selected.id != "struggle":
                cost = 1 + int(not defender.is_fainted and clean_key(defender.ability) == "pressure" and selected.target in (
                    "normal", "allAdjacentFoes", "allAdjacent", "any", "randomNormal", "foeSide"))
                selected.pp = max(0, selected.pp - cost)

            if sleep_talk_failed:
                moved_players.add(player)
                continue

            if defender.is_fainted and move.target in ("normal", "allAdjacentFoes", "allAdjacent", "any", "randomNormal"):
                # A target can disappear via recoil before this queued attempt.
                # The attempt still consumes PP; self/field moves can still act.
                moved_players.add(player)
                continue

            # These moves require a damaging move still queued for this target.
            if m_id in ("suckerpunch", "thunderclap"):
                opp_move = m2 if player == "p1" else m1
                opp_player = "p2" if player == "p1" else "p1"
                if opp_move is None or (opp_move.category == MoveCategory.STATUS and opp_move.id != "mefirst") or opp_player in moved_players or "mustrecharge" in defender.volatiles:
                    moved_players.add(player)
                    continue

            battle_weather = effective_weather(s.weather,attacker,defender)
            # PrepareHit runs after the move's Try check, before Protect/accuracy.
            if clean_key(attacker.ability) == "protean" and not attacker.protean_used and not attacker.is_terastallized and m_id != "struggle":
                kind = resolved_move_type(attacker,move,battle_weather,s.terrain)
                if attacker.active_types != (kind,None) and kind != PokemonType.STELLAR:
                    attacker.type_override = (kind,None)
                    attacker.protean_used = True

            # 1. Protection moves
            if getattr(move, "is_protect", False) or m_id in ("protect", "spikyshield", "detect", "banefulbunker", "silktrap"):
                protect_count = getattr(attacker, "protect_streak", 0)
                # Authentic Gen 9 Protect success decay formula: 1 / 3^k
                success_rate = 1.0 / (3.0 ** min(protect_count, 6))
                if sample_outcomes:
                    success_rate = float(rng.random() < success_rate)
                    if not any(who != player and who not in moved_players for who, _ in order):
                        success_rate = 0.0  # Protect fails when no action remains to block.
                attacker.protect_success_rate = success_rate
                attacker.protect_streak = protect_count + 1 if success_rate else 0
                attacker.last_protect_move = m_id
                moved_players.add(player)
                continue
            else:
                attacker.protect_streak = 0
                attacker.protect_success_rate = 0.0

            succ_rate = getattr(defender, "protect_success_rate", 0.0)
            targets_foe = move.target in ("normal", "allAdjacentFoes", "allAdjacent", "any", "randomNormal")
            if sample_outcomes and succ_rate == 1.0 and targets_foe and move.blocked_by_protect:
                if getattr(defender, "last_protect_move", "") == "spikyshield" and move.is_contact:
                    attacker.take_damage(max(1, attacker.max_hp // 8))
                moved_players.add(player)
                continue

            battle_weather = effective_weather(s.weather,attacker,defender)
            if targets_foe and move.category==MoveCategory.STATUS and clean_key(attacker.ability)=="prankster" and PokemonType.DARK in defender.active_types:
                moved_players.add(player)
                continue
            if targets_foe and s.terrain==Terrain.PSYCHIC and defender.is_grounded() and execution_priority>0:
                moved_players.add(player)
                continue
            # Gen 9 TryHit absorption precedes accuracy and Substitute.
            damage_type = resolved_move_type(attacker,move,battle_weather,s.terrain)
            if targets_foe and clean_key(defender.ability) == "waterabsorb" and damage_type == PokemonType.WATER:
                defender.heal(max(1,defender.max_hp//4))
                moved_players.add(player)
                continue
            if targets_foe and clean_key(defender.ability) == "flashfire" and damage_type == PokemonType.FIRE:
                defender.volatiles.setdefault("flashfire",{})
                moved_players.add(player)
                continue
            if sample_outcomes and targets_foe and rng.random() >= accuracy_chance(attacker,defender,move,battle_weather):
                moved_players.add(player)
                continue

            if move.category == MoveCategory.STATUS and targets_foe and clean_key(defender.ability) == "goodasgold":
                moved_players.add(player)
                continue
            hits_substitute = ("substitute" in defender.volatiles and targets_foe
                               and not move.bypass_substitute and not move.is_sound
                               and clean_key(attacker.ability) != "infiltrator")
            if hits_substitute and move.category == MoveCategory.STATUS:
                moved_players.add(player)
                continue
            actual_dmg = 0
            # 2. Damage calculation
            if move.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                total_damage = 0
                for hit in range(1,hit_count(attacker,move,rng,sample_outcomes)+1):
                    if attacker.is_fainted or defender.is_fainted:break
                    if hit > 1 and move.multiaccuracy and clean_key(attacker.ability) != "skilllink" and clean_key(attacker.item) != "loadeddice":
                        if sample_outcomes and rng.random() >= accuracy_chance(attacker,defender,move,battle_weather):break
                    hits_substitute = ("substitute" in defender.volatiles and targets_foe
                                       and not move.bypass_substitute and not move.is_sound
                                       and clean_key(attacker.ability) != "infiltrator")
                    hit_move = move
                    if m_id in ("tripleaxel","triplekick"):
                        hit_move = move.clone()
                        hit_move.base_power = (20 if m_id == "tripleaxel" else 10)*hit
                    atk_side = s.p1 if player == "p1" else s.p2
                    critical = False
                    if sample_outcomes and move.base_power > 0:
                        ratio = move.crit_ratio + int(clean_key(attacker.ability) == "superluck")
                        ratio += int(clean_key(attacker.item) in ("scopelens", "razorclaw"))
                        denominator = (24, 8, 2, 1)[max(0, min(3, ratio - 1))]
                        critical = move.will_crit or rng.randrange(denominator) == 0
                        if clean_key(defender.ability) in ("battlearmor", "shellarmor"):
                            critical = False
                    rolls = calculate_damage_rolls(
                        attacker, defender, hit_move, battle_weather, s.terrain,
                        attacker_side=atk_side, is_critical=critical,
                        defender_side=s.p2 if player == "p1" else s.p1,
                    )
                    median_dmg = rng.choice(rolls) if sample_outcomes else rolls[len(rolls) // 2]
                    # Sampled blocks returned above; moves that bypass Protect reach here.
                    succ_rate = 0.0 if sample_outcomes else getattr(defender, "protect_success_rate", 0.0)
                    is_contact = getattr(move, "is_contact", False) or is_contact_move(m_id, move.category)

                    if succ_rate > 0.0:
                        # Defender takes damage proportional to protect failure rate
                        actual_dmg = int(median_dmg * (1.0 - succ_rate))
                        actual_dmg = defender.take_damage(actual_dmg)
                        # Spiky shield chip on physical/special contact when protected
                        if succ_rate > 0.5 and getattr(defender, "last_protect_move", "") == "spikyshield" and is_contact:
                            attacker.take_damage(max(1, attacker.max_hp // 8))
                    else:
                        if hits_substitute:
                            sub = defender.volatiles["substitute"]
                            # Public logs hide exact remaining substitute HP. The local
                            # search hypothesis starts with one quarter, marked unknown.
                            sub_hp = sub.get("hp", -1)
                            if sub_hp < 0: sub_hp = max(1,defender.max_hp // 4)
                            actual_dmg = min(sub_hp, median_dmg)
                            sub["hp"] = sub_hp - actual_dmg
                            if sub["hp"] <= 0: defender.volatiles.pop("substitute")
                        else:
                            actual_dmg = defender.take_damage(median_dmg)
                        # Rocky Helmet chip on contact when attack connects
                        def_it = clean_key(defender.item)
                        if not hits_substitute and actual_dmg > 0 and def_it == "rockyhelmet" and is_contact and clean_key(attacker.ability) != "magicguard":
                            attacker.take_damage(max(1, attacker.max_hp // 6))

                    if not hits_substitute and actual_dmg and defender.status == StatusCondition.FREEZE and move.move_type == PokemonType.FIRE:
                        defender.status = StatusCondition.NONE
                    if m_id == "struggle":
                        attacker.take_damage(max(1, (attacker.max_hp + 2) // 4))

                    # Pop Air Balloon on direct damage
                    if actual_dmg > 0 and clean_key(defender.item) == "airballoon":
                        defender.item = None

                    # Knock Off item removal (if attack lands and defender not protected)
                    if not hits_substitute and actual_dmg > 0 and m_id == "knockoff" and succ_rate <= 0.5 and defender.item:
                        if is_removable_item(defender.item):
                            defender.item = None

                    # Drain moves: heal attacker by exact canonical ratio or default 50%
                    if actual_dmg > 0 and getattr(move, "drain", None):
                        num, den = move.drain
                        drain_heal = max(1, (actual_dmg * num + den - 1) // den)
                        attacker.heal(drain_heal)
                    elif actual_dmg > 0 and m_id in ("hornleech", "drainpunch", "gigadrain", "absorb", "megadrain", "drainingkiss", "oblivionwing", "bitterblade", "paraboliccharge"):
                        drain_heal = max(1, actual_dmg // 2)
                        attacker.heal(drain_heal)

                    # Recoil moves: recoil by exact canonical ratio or standard fraction
                    if actual_dmg > 0 and clean_key(attacker.ability) not in ("rockhead","magicguard") and getattr(move, "recoil", None):
                        num, den = move.recoil
                        attacker.take_damage(max(1, (2*actual_dmg*num+den)//(2*den)))
                    elif actual_dmg > 0 and clean_key(attacker.ability) not in ("rockhead","magicguard") and m_id in ("bravebird", "flareblitz", "woodhammer", "wavecrash", "doubleedge"):
                        attacker.take_damage(max(1, actual_dmg // 3))
                    elif actual_dmg > 0 and clean_key(attacker.ability) not in ("rockhead","magicguard") and m_id in ("headsmash",):
                        attacker.take_damage(max(1, actual_dmg // 2))

                    def_ab = clean_key(defender.ability)
                    # Secondary effects occur on each connected strike.
                    if actual_dmg > 0 and clean_key(attacker.ability) != "sheerforce":
                        for sec in move.secondaries:
                            chance = min(100,sec.get("chance",100)*(2 if clean_key(attacker.ability)=="serenegrace" else 1))
                            triggered = (rng.random()*100 < chance) if sample_outcomes and chance < 100 else chance >= 100
                            if not triggered:
                                continue
                            for key,delta in sec.get("self",{}).get("boosts",{}).items():
                                if not attacker.is_fainted:
                                    attacker.boosts[key] = max(-6,min(6,attacker.boosts.get(key,0)+delta))
                            if hits_substitute or defender.is_fainted or def_ab == "shielddust" or clean_key(defender.item)=="covertcloak":
                                continue
                            for key,delta in sec.get("boosts",{}).items():
                                defender.boosts[key] = max(-6,min(6,defender.boosts.get(key,0)+delta))
                            if sec.get("volatileStatus") == "flinch" and def_ab != "innerfocus":
                                other = "p2" if player=="p1" else "p1"
                                if other not in moved_players:
                                    flinched.add(other)
                            if sec.get("volatileStatus") == "confusion" and def_ab != "owntempo" and not (defender.is_grounded() and s.terrain == Terrain.MISTY):
                                defender.volatiles.setdefault("confusion", {"time":rng.randint(2,5) if sample_outcomes else 3})
                            if defender.status == StatusCondition.NONE:
                                kind = sec.get("status")
                                immune = ((kind=="brn" and (PokemonType.FIRE in defender.active_types or def_ab in ("waterveil","waterbubble"))) or
                                          (kind in ("psn","tox") and (PokemonType.POISON in defender.active_types or PokemonType.STEEL in defender.active_types or def_ab=="immunity")) or
                                          (kind=="par" and (PokemonType.ELECTRIC in defender.active_types or def_ab=="limber")) or
                                          (kind=="frz" and (PokemonType.ICE in defender.active_types or def_ab=="magmaarmor" or s.weather in (Weather.SUN,Weather.HARSH_SUN))) or
                                          (defender.is_grounded() and s.terrain==Terrain.MISTY))
                                if kind and not immune:
                                    statuses={"brn":StatusCondition.BURN,"par":StatusCondition.PARALYSIS,"psn":StatusCondition.POISON,"tox":StatusCondition.TOXIC,"frz":StatusCondition.FREEZE}
                                    if kind in statuses:
                                        defender.status=statuses[kind]

                    consume_lum_berry(defender)
                    total_damage += actual_dmg
                    if actual_dmg == 0:break
                actual_dmg = total_damage

                # Life Orb recoil (10% max HP)
                if actual_dmg > 0 and clean_key(attacker.item) == "lifeorb" and clean_key(attacker.ability) != "magicguard" and not (clean_key(attacker.ability)=="sheerforce" and move.secondaries):
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
            if getattr(move, "boosts", None) and not hits_substitute:
                is_blocked = (succ_rate > 0.5) or (move.category == MoveCategory.STATUS and def_ab == "goodasgold" and any(v < 0 for v in move.boosts.values()))
                if not is_blocked:
                    for stat_k, boost_val in move.boosts.items():
                        curr_b = defender.boosts.get(stat_k, 0)
                        defender.boosts[stat_k] = max(-6, min(6, curr_b + boost_val))

            if m_id == "substitute" and "substitute" not in attacker.volatiles and attacker.current_hp > attacker.max_hp / 4 and attacker.max_hp > 1:
                cost = attacker.max_hp // 4
                attacker.take_damage(cost)
                attacker.volatiles["substitute"] = {"hp":cost}
                attacker.volatiles.pop("partiallytrapped",None)
            elif m_id in ("encore","disable"):
                target = attacker if is_magic_bounce else defender
                previous = next((m for m in target.moves if m.id == target.last_move),None)
                allowed = previous and previous.pp > 0 and previous.id != "struggle"
                if m_id == "encore" and previous and previous.fail_encore:allowed = False
                if allowed and m_id not in target.volatiles and clean_key(target.ability) not in ("aromaveil","goodasgold"):
                    if clean_key(target.item) == "mentalherb":
                        target.item = None
                        for condition in ("taunt","encore","disable","torment","healblock","attract"):
                            target.volatiles.pop(condition,None)
                    else:
                        other = "p2" if player == "p1" else "p1"
                        has_queued_move = not is_magic_bounce and other not in moved_players and selected_users.get(other) is target
                        duration = (3 if m_id == "encore" else 4) + int(not has_queued_move)
                        target.volatiles[m_id] = {"duration":duration,"move":previous.id}
            elif m_id == "leechseed":
                target = attacker if is_magic_bounce else defender
                if PokemonType.GRASS not in target.active_types and clean_key(target.ability) != "goodasgold":
                    target.volatiles.setdefault("leechseed", {"source_side":(2 if player == "p1" else 1) if is_magic_bounce else (1 if player == "p1" else 2)})
            elif m_id == "taunt":
                target = attacker if is_magic_bounce else defender
                if clean_key(target.ability) not in ("oblivious","aromaveil","goodasgold"):
                    if clean_key(target.item) == "mentalherb":
                        target.item = None
                    else:
                        other = player if is_magic_bounce else ("p2" if player == "p1" else "p1")
                        # A late Taunt includes three *future* turns after this residual.
                        target.volatiles.setdefault("taunt", {"duration":4 if is_magic_bounce or (other in moved_players and selected_users.get(other) is target) else 3})
            elif move.volatile_status == "confusion":
                target = attacker if is_magic_bounce else defender
                if clean_key(target.ability) != "owntempo" and not (target.is_grounded() and s.terrain == Terrain.MISTY):
                    target.volatiles.setdefault("confusion", {"time":rng.randint(2,5) if sample_outcomes else 3})
            elif (move.volatile_status in ("partiallytrapped", "trapped") or m_id in ("meanlook","block","spiderweb")) and not hits_substitute and not defender.is_fainted:
                if move.category == MoveCategory.STATUS or actual_dmg > 0:
                    source_side = 1 if player == "p1" else 2
                    source = (source_side,(s.p1 if source_side == 1 else s.p2).active_index)
                    if move.volatile_status == "partiallytrapped":
                        duration = 8 if clean_key(attacker.item)=="gripclaw" else (rng.randint(5,6) if sample_outcomes else 5)
                        defender.volatiles.setdefault("partiallytrapped", {"duration":duration,"source":source,
                            "divisor":6 if clean_key(attacker.item)=="bindingband" else 8})
                    else:
                        target = attacker if is_magic_bounce else defender
                        if PokemonType.GHOST not in target.active_types:
                            if is_magic_bounce:
                                source_side = 3-source_side
                                source = (source_side,(s.p1 if source_side == 1 else s.p2).active_index)
                            target.volatiles.setdefault("trapped", {"source":source})

            if m_id in WEATHER_MOVES:set_weather(s,WEATHER_MOVES[m_id],attacker)
            if m_id in TERRAIN_MOVES:set_terrain(s,TERRAIN_MOVES[m_id],attacker)

            # Side/field effects are public and have finite cartridge durations.
            user_side = s.p1 if player == "p1" else s.p2
            if m_id in ("reflect","lightscreen","auroraveil"):
                if m_id != "auroraveil" or s.weather == Weather.SNOW:
                    if not user_side.screens.get(m_id):
                        user_side.screens[m_id] = 8 if clean_key(attacker.item) == "lightclay" else 5
            elif m_id == "tailwind":
                if not user_side.tailwind:
                    user_side.tailwind = 4
            elif m_id == "trickroom":
                s.trick_room = 0 if s.trick_room else 5

            # Additional field / status mechanics
            if m_id in ("rapidspin", "mortalspin", "tidyup"):
                is_blocked = (succ_rate > 0.5 or actual_dmg == 0) if m_id != "tidyup" else False
                if not is_blocked:
                    user_side = s.p1 if player == "p1" else s.p2
                    user_side.hazards.clear()
                    attacker.volatiles.pop("partiallytrapped",None)
                    if m_id == "rapidspin":attacker.volatiles.pop("leechseed",None)
                    if m_id == "tidyup":
                        s.p1.hazards.clear()
                        s.p2.hazards.clear()
                        attacker.volatiles.pop("substitute",None)
                        defender.volatiles.pop("substitute",None)
            elif m_id == "courtchange":
                s.p1.hazards, s.p2.hazards = s.p2.hazards, s.p1.hazards
                s.p1.screens, s.p2.screens = s.p2.screens, s.p1.screens
                s.p1.tailwind, s.p2.tailwind = s.p2.tailwind, s.p1.tailwind
            elif m_id == "icespinner" and actual_dmg > 0:
                s.terrain = Terrain.NONE
                s.terrain_turns = 0
            elif m_id == "ceaselessedge" and actual_dmg > 0:
                opp_side = s.p2 if player == "p1" else s.p1
                opp_side.hazards[Hazard.SPIKES_1] = min(3, opp_side.hazards.get(Hazard.SPIKES_1, 0) + 1)
            elif m_id == "stoneaxe" and actual_dmg > 0:
                opp_side = s.p2 if player == "p1" else s.p1
                opp_side.hazards[Hazard.STEALTH_ROCK] = 1
            elif m_id == "stealthrock":
                target_side = (s.p1 if player == "p1" else s.p2) if is_magic_bounce else (s.p2 if player == "p1" else s.p1)
                target_side.hazards[Hazard.STEALTH_ROCK] = 1
            elif m_id == "spikes":
                target_side = (s.p1 if player == "p1" else s.p2) if is_magic_bounce else (s.p2 if player == "p1" else s.p1)
                target_side.hazards[Hazard.SPIKES_1] = min(3, target_side.hazards.get(Hazard.SPIKES_1, 0) + 1)
            elif m_id == "defog":
                target = attacker if is_magic_bounce else defender
                source = defender if is_magic_bounce else attacker
                if "substitute" not in target.volatiles or clean_key(source.ability) == "infiltrator":
                    target.boosts["evasion"] = max(-6,target.boosts.get("evasion",0)-1)
                s.p1.hazards.clear()
                s.p2.hazards.clear()
                (s.p1 if (player=="p1")==is_magic_bounce else s.p2).screens.clear()
                s.terrain = Terrain.NONE
                s.terrain_turns = 0
            elif m_id in ("uturn", "voltswitch", "flipturn"):
                user_side = s.p1 if player == "p1" else s.p2
                if succ_rate <= 0.5 and actual_dmg > 0 and not attacker.is_fainted and user_side.available_switches() and not s.is_game_over:
                    s.pending_switches = (1 if player == "p1" else 2,)
            elif m_id in ("whirlwind", "roar", "dragontail", "circlethrow"):
                if (move.category != MoveCategory.STATUS and (actual_dmg == 0 or hits_substitute)) or defender.is_fainted or def_ab == "suctioncups":
                    pass
                elif m_id in ("whirlwind", "roar") and def_ab == "goodasgold":
                    pass  # Good as Gold is immune to status phazing
                elif is_magic_bounce and m_id in ("whirlwind", "roar"):
                    # Magic Bounce reflects phazing back to user side
                    u_side = s.p1 if player == "p1" else s.p2
                    u_sw = u_side.available_switches()
                    if u_sw:
                        perform_switch(s,1 if player == "p1" else 2,rng.choice(u_sw) if sample_outcomes else u_sw[0])
                        if player == "p1":
                            p1_active = s.p1.active_pokemon
                        else:
                            p2_active = s.p2.active_pokemon
                else:
                    opp_side = s.p2 if player == "p1" else s.p1
                    opp_sw = opp_side.available_switches()
                    if opp_sw:
                        opp_target = rng.choice(opp_sw) if sample_outcomes else opp_sw[0]
                        perform_switch(s,2 if player == "p1" else 1,opp_target)
                        if player == "p1":
                            p2_active = s.p2.active_pokemon
                        else:
                            p1_active = s.p1.active_pokemon
            elif (getattr(move, "is_heal", False) and move.category == MoveCategory.STATUS and m_id != "rest") or m_id in ("recover", "roost", "slackoff", "softboiled", "wish", "synthesis", "moonlight", "morningsun", "shoreup", "milkdrink", "healorder"):
                if attacker.current_hp < attacker.max_hp:
                    divisor = 2
                    if m_id in ("synthesis","moonlight","morningsun"):
                        if battle_weather in (Weather.SUN,Weather.HARSH_SUN):
                            attacker.heal((attacker.max_hp*2+1)//3)
                            divisor = 0
                        elif battle_weather != Weather.NONE:divisor = 4
                    if divisor:
                        rounded = m_id not in ("synthesis","moonlight","morningsun","shoreup")
                        attacker.heal((attacker.max_hp + (divisor//2 if rounded else 0)) // divisor)
                    if m_id == "roost" and not attacker.is_terastallized:
                        attacker.volatiles["roost"] = {"duration":1}
            elif m_id == "rest":
                if attacker.current_hp < attacker.max_hp and attacker.status != StatusCondition.SLEEP and not (attacker.is_grounded() and s.terrain in (Terrain.ELECTRIC,Terrain.MISTY)):
                    attacker.heal(attacker.max_hp)
                    attacker.status = StatusCondition.SLEEP
                    attacker.status_turns = 3
            elif m_id == "willowisp":
                target_mon = attacker if is_magic_bounce else defender
                t_ab = clean_key(target_mon.ability)
                if PokemonType.FIRE not in target_mon.active_types and t_ab != "goodasgold" and target_mon.status == StatusCondition.NONE and not (target_mon.is_grounded() and s.terrain == Terrain.MISTY):
                    target_mon.status = StatusCondition.BURN
            elif m_id == "toxic":
                target_mon = attacker if is_magic_bounce else defender
                t_ab = clean_key(target_mon.ability)
                if PokemonType.POISON not in target_mon.active_types and PokemonType.STEEL not in target_mon.active_types and t_ab != "goodasgold" and target_mon.status == StatusCondition.NONE and not (target_mon.is_grounded() and s.terrain == Terrain.MISTY):
                    target_mon.status = StatusCondition.TOXIC
            elif m_id == "thunderwave":
                target_mon = attacker if is_magic_bounce else defender
                t_ab = clean_key(target_mon.ability)
                if PokemonType.ELECTRIC not in target_mon.active_types and PokemonType.GROUND not in target_mon.active_types and t_ab != "goodasgold" and target_mon.status == StatusCondition.NONE and not (target_mon.is_grounded() and s.terrain == Terrain.MISTY):
                    target_mon.status = StatusCondition.PARALYSIS

            consume_lum_berry(attacker)
            consume_lum_berry(defender)
            moved_players.add(player)
            sync_paradox(s)
            if s.pending_switches:
                remaining = []
                for who, queued_move in order[order_index+1:]:
                    user = selected_users[who]
                    slot = next((i for i,m in enumerate(user.moves) if m is queued_move), -1)
                    remaining.append((who, slot, queued_move.clone()))
                s.continuation = dict(order=remaining, moved=list(moved_players), flinched=list(flinched),
                    users={who:next(i for i,p in enumerate((s.p1 if who == "p1" else s.p2).pokemon) if p is mon)
                           for who,mon in selected_users.items() if mon}, m1=m1, m2=m2)
                return finalized_state(s)

    if s.is_game_over:
        return finalized_state(s)  # Showdown ends immediately; no residual damage/healing or duration tick.

    weather_residual(s)
    if s.is_game_over:return finalized_state(s)

    # Phase 4: End-of-turn effects (Leftovers, Black Sludge, Poison Heal, Status Orbs, Burn, Poison)
    for act_mon in (s.p1.active_pokemon, s.p2.active_pokemon):
        if act_mon and not act_mon.is_fainted:
            ab = clean_key(act_mon.ability)
            it = clean_key(act_mon.item)

            # Leftovers
            if it == "leftovers":
                act_mon.heal(max(1, act_mon.max_hp // 16))

            if s.terrain==Terrain.GRASSY and act_mon.is_grounded():
                act_mon.heal(max(1,act_mon.max_hp//16))

            # Black Sludge
            if it == "blacksludge":
                if PokemonType.POISON in act_mon.active_types:
                    act_mon.heal(max(1, act_mon.max_hp // 16))
                else:
                    act_mon.take_damage(max(1, act_mon.max_hp // 8))

    if s.is_game_over:return finalized_state(s)

    # Leech Seed follows the source's active slot, not the original Pokemon.
    seed_sides = [side for side in (s.p1,s.p2) if side.active_pokemon]
    seed_sides.sort(key=lambda side: speed_order_key(side.active_pokemon,side,s),reverse=True)
    for side in seed_sides:
        seeded = side.active_pokemon
        source_side = seeded.volatiles.get("leechseed",{}).get("source_side") if seeded else None
        recipient = (s.p1 if source_side == 1 else s.p2).active_pokemon if source_side else None
        if seeded and not seeded.is_fainted and recipient and not recipient.is_fainted and clean_key(seeded.ability) != "magicguard":
            drained = seeded.take_damage(max(1,seeded.max_hp//8))
            if clean_key(seeded.ability) == "liquidooze":
                if clean_key(recipient.ability) != "magicguard":recipient.take_damage(drained)
            else:
                healed = (drained*5324+2047)//4096 if clean_key(recipient.item) == "bigroot" else drained
                recipient.heal(healed)
            if s.is_game_over:return finalized_state(s)

    for act_mon in (s.p1.active_pokemon,s.p2.active_pokemon):
        if act_mon and not act_mon.is_fainted:
            ab,it = clean_key(act_mon.ability),clean_key(act_mon.item)
            # Poison / Toxic (with Poison Heal support)
            if act_mon.status in (StatusCondition.POISON, StatusCondition.TOXIC):
                if ab == "poisonheal":
                    act_mon.heal(max(1, act_mon.max_hp // 8))
                elif ab != "magicguard":
                    if act_mon.status == StatusCondition.TOXIC:
                        act_mon.toxic_counter = min(15,getattr(act_mon,"toxic_counter",0)+1)
                        act_mon.take_damage(max(1, act_mon.max_hp // 16)*act_mon.toxic_counter)
                    else:
                        act_mon.take_damage(max(1, act_mon.max_hp // 8))
            elif act_mon.status == StatusCondition.BURN and ab != "magicguard":
                act_mon.take_damage(max(1, act_mon.max_hp // 16))

            trap = act_mon.volatiles.get("partiallytrapped")
            if trap:
                source = trap.get("source")
                source_mon = (s.p1 if source[0] == 1 else s.p2).pokemon[source[1]] if source else None
                source_active = not source or ((s.p1 if source[0] == 1 else s.p2).active_index == source[1] and not source_mon.is_fainted)
                if trap.get("duration",-1) > 0: trap["duration"] -= 1
                if not source_active or trap.get("duration") == 0:
                    act_mon.volatiles.pop("partiallytrapped")
                elif ab != "magicguard":
                    act_mon.take_damage(max(1,act_mon.max_hp // trap.get("divisor",8)))
            for condition in ("encore","disable"):
                effect = act_mon.volatiles.get(condition)
                if effect:
                    if effect.get("duration",-1) > 0:effect["duration"] -= 1
                    exhausted = condition == "encore" and not any(m.id == effect.get("move") and m.pp > 0 for m in act_mon.moves)
                    if effect.get("duration") == 0 or exhausted:act_mon.volatiles.pop(condition)
            taunt = act_mon.volatiles.get("taunt")
            if taunt and taunt.get("duration",-1) > 0:
                taunt["duration"] -= 1
                if not taunt["duration"]: act_mon.volatiles.pop("taunt")

            # Status Orbs
            if not act_mon.is_fainted and act_mon.status == StatusCondition.NONE:
                if it == "flameorb" and PokemonType.FIRE not in act_mon.active_types:
                    act_mon.status = StatusCondition.BURN
                elif it == "toxicorb" and PokemonType.POISON not in act_mon.active_types and PokemonType.STEEL not in act_mon.active_types:
                    act_mon.status = StatusCondition.TOXIC



    if s.is_game_over:return finalized_state(s)
    for side in (s.p1,s.p2):
        if side.active_pokemon:side.active_pokemon.volatiles.pop("roost",None)
        side.tailwind = max(0,side.tailwind-1)
        side.screens = {name:(turns-1 if turns>0 else turns) for name,turns in side.screens.items() if turns != 1}
    s.trick_room = max(0,s.trick_room-1)
    if s.terrain_turns > 0:
        s.terrain_turns -= 1
        if s.terrain_turns == 0:s.terrain = Terrain.NONE
    sync_paradox(s)
    s.pending_switches = tuple(i for i,side in ((1,s.p1),(2,s.p2))
        if side.active_pokemon and side.active_pokemon.is_fainted and not side.is_all_fainted)
    if not s.pending_switches and not s.is_game_over:
        s.turn += 1

    return finalized_state(s)


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

    def _resolve_replacements(self, state, depth, sample, return_both_players=False):
        """Only the requested sides choose; queued attacks resume on that choice.

        A replacement does not consume another search turn. Simultaneous faint
        replacements form a switch-only matrix; a single replacement is max/min.
        """
        actions1 = state.get_valid_actions(1) if 1 in state.pending_switches else [None]
        actions2 = state.get_valid_actions(2) if 2 in state.pending_switches else [None]
        matrix = np.zeros((len(actions1),len(actions2)))
        leaves,leaf_coords = [],[]
        for i,a1 in enumerate(actions1):
            for j,a2 in enumerate(actions2):
                child = simulate_turn_transition(state,a1,a2)
                if child.is_game_over:
                    value = 15.0 if child.winner == 1 else (-15.0 if child.winner == 2 else 0.0)
                elif child.pending_switches:
                    value = self._resolve_replacements(child,depth,False)[3]
                elif depth > 0:
                    value = self.resolve_turn(child,depth=depth,sample=False)[3]
                else:
                    leaves.append(child)
                    leaf_coords.append((i,j))
                    continue
                matrix[i,j] = value
        if leaves:
            values = (self.evaluator.evaluate_batch(leaves) if hasattr(self.evaluator,"evaluate_batch")
                      else [self.evaluator.evaluate(child) for child in leaves])
            for (i,j),value in zip(leaf_coords,values):
                matrix[i,j] = value
        pi1,pi2,value = solve_zero_sum_game(matrix)
        i = int(np.random.choice(len(pi1),p=pi1)) if sample else int(np.argmax(pi1))
        j = int(np.random.choice(len(pi2),p=pi2)) if sample else int(np.argmax(pi2))
        result = (actions1[i],pi1 if actions1[0] is not None else np.array([]),
                  actions1 if actions1[0] is not None else [],float(value))
        if return_both_players:
            return result + (actions2[j],pi2 if actions2[0] is not None else np.array([]),
                             actions2 if actions2[0] is not None else [])
        return result

    def resolve_turn(
        self,
        state: BattleState,
        depth: int = 1,
        sample: bool = True,
        p1_actions_override: Optional[List[Action]] = None,
        p1_sucker_streak: int = 0,
        last_action_was_switch: bool = False,
        cand_beam: Optional[Tuple[int, int]] = None,
        is_leaf_eval: bool = False,
        return_both_players: bool = False
    ) -> Tuple:
        """Compute the Nash Equilibrium mixed strategy for the current turn.

        Returns:
            chosen_action: The selected action (sampled from mixed strategy or argmax).
            p1_strategy: The full Nash probability distribution over p1_actions.
            p1_actions: List of valid actions for P1.
            expected_value: The game-theoretic value of the position.
            (optional if return_both_players=True): chosen_action_p2, p2_strategy, p2_actions
        """
        if state.is_game_over:
            val = 1.0 if state.winner == 1 else -1.0
            if return_both_players:
                return (None, np.array([]), [], val, None, np.array([]), [])
            return (None, np.array([]), [], val)

        if state.pending_switches:
            return self._resolve_replacements(state,depth,sample,return_both_players)

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

        spe1_before = speed_order_key(p1_mon_before, state.p1, state) if p1_mon_before and not p1_mon_before.is_fainted else 0
        spe2_before = speed_order_key(p2_mon_before, state.p2, state) if p2_mon_before and not p2_mon_before.is_fainted else 0

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
                    if m1_obj and m2_obj and move_priority(p1_mon_before,m1_obj) == move_priority(p2_mon_before,m2_obj) and spe1_before == spe2_before:
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
            # Replacement nodes must resolve their remaining turn before evaluation.
            batch_indices = [i for i,child in enumerate(states_to_eval) if not child.pending_switches]
            batch_states = [states_to_eval[i] for i in batch_indices]
            values = {}
            if batch_states:
                batch_values = (self.evaluator.evaluate_batch(batch_states) if hasattr(self.evaluator,"evaluate_batch")
                                else [self.evaluator.evaluate(child) for child in batch_states])
                values.update(zip(batch_indices,batch_values))
            for i,child in enumerate(states_to_eval):
                if child.pending_switches:
                    values[i] = self._resolve_replacements(child,0,False)[3]
            for k,(i,j,tie_idx) in enumerate(eval_coords):
                if tie_idx == 0:
                    base_M[i,j] = float(values[k])
                else:
                    base_M[i,j] += .5 * float(values[k])

        M = base_M.copy()

        # Post-transition adjustments: Material Faints, Setup Blunders, Switch Costs
        # Pre-compute whether active Pokémon hold guaranteed 1-hit KO moves or threats on their active target
        p1_has_lethal = False
        p1_has_faster_lethal = False
        if p1_mon_before and p2_mon_before and not p1_mon_before.is_fainted and not p2_mon_before.is_fainted:
            spe1 = speed_order_key(p1_mon_before, state.p1, state)
            spe2 = speed_order_key(p2_mon_before, state.p2, state)
            for a in p1_actions:
                if a.action_type == ActionType.MOVE:
                    mv = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a.move_id), None)
                    if mv and mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                        rolls = calculate_damage_rolls(p1_mon_before, p2_mon_before, mv, state.weather, state.terrain, attacker_side=state.p1, defender_side=state.p2)
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
            spe1 = speed_order_key(p1_mon_before, state.p1, state)
            spe2 = speed_order_key(p2_mon_before, state.p2, state)
            for a in p2_actions:
                if a.action_type == ActionType.MOVE:
                    mv = next((x for x in p2_mon_before.moves if getattr(x, "id", "") == a.move_id), None)
                    if mv and mv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                        rolls = calculate_damage_rolls(p2_mon_before, p1_mon_before, mv, state.weather, state.terrain, attacker_side=state.p2, defender_side=state.p1)
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
                    d_rolls = calculate_damage_rolls(p1_mon_before, p2_mon_before, dm, state.weather, state.terrain, attacker_side=state.p1, defender_side=state.p2)
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
                    spe1 = speed_order_key(p1_mon_before, state.p1, state)
                    spe_col = speed_order_key(col_def_mon, state.p2, state)
                    for ca in p1_actions:
                        if ca.action_type == ActionType.MOVE:
                            cmv = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == ca.move_id), None)
                            if cmv and cmv.category in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                                crolls = calculate_damage_rolls(p1_mon_before, col_def_mon, cmv, state.weather, state.terrain, attacker_side=state.p1, defender_side=state.p2)
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
                            rolls = calculate_damage_rolls(p1_mon_before, col_def_mon, m1_obj, state.weather, state.terrain, attacker_side=state.p1, defender_side=state.p2)
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
                            rolls = calculate_damage_rolls(p1_mon_before, col_def_mon, m1_obj, state.weather, state.terrain, attacker_side=state.p1, defender_side=state.p2)
                            if not rolls or min(rolls) < col_def_mon.current_hp:
                                M[i, j] -= 6.00  # Continuing to attack at -2/-4/-6 SpA without KO is strictly penalized!

                    if (is_p1_debuffed or p2_has_faster_lethal) and is_p1_threatened and not col_p1_faster_lethal:
                        # Threatened mon facing faster lethal cannot kill opponent: do NOT throw it away for futile move
                        m1_obj = next((x for x in p1_mon_before.moves if getattr(x, "id", "") == a1.move_id), None)
                        if m1_obj:
                            rolls = calculate_damage_rolls(p1_mon_before, col_def_mon, m1_obj, state.weather, state.terrain, attacker_side=state.p1, defender_side=state.p2)
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
                        rolls = calculate_damage_rolls(p1_mon_before, col_def_mon, m1_obj, state.weather, state.terrain, attacker_side=state.p1, defender_side=state.p2)
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
                            rolls = calculate_damage_rolls(p2_mon_before, p1_mon_before, m2_mv, state.weather, state.terrain, attacker_side=state.p2, defender_side=state.p1)
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
                            rolls = calculate_damage_rolls(p1_mon_before, p2_mon_before, m1_mv, state.weather, state.terrain, attacker_side=state.p1, defender_side=state.p2)
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
                        sp1_rolls = calculate_damage_rolls(p1_mon_before, p2_mon_before, sp1_mv, state.weather, state.terrain, attacker_side=state.p1, defender_side=state.p2)
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
                        sp2_rolls = calculate_damage_rolls(p2_mon_before, p1_mon_before, sp2_mv, state.weather, state.terrain, attacker_side=state.p2, defender_side=state.p1)
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

            child_beam = (2, 2)
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

        if return_both_players:
            if sample:
                chosen_idx2 = np.random.choice(m, p=p2_strat)
            else:
                chosen_idx2 = int(np.argmax(p2_strat))
            return p1_actions[chosen_idx], p1_strat, p1_actions, game_value, p2_actions[chosen_idx2], p2_strat, p2_actions

        return p1_actions[chosen_idx], p1_strat, p1_actions, game_value
