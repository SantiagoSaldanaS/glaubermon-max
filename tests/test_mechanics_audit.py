"""Comprehensive mechanics audit test suite for Gen 9 OU engine."""

import pytest
from glaubermon.core.types import PokemonType, MoveCategory, Weather, StatusCondition, Hazard, ActionType
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.actions import MoveAction, SwitchAction, action_to_logit_index
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.inference.damage_calc import calculate_damage_rolls
from glaubermon.data.meta_teams import get_meta_team_balance, build_meta_pokemon
from glaubermon.search.subgame_resolver import simulate_turn_transition, SubgameResolver
from glaubermon.search.evaluators import HeuristicEvaluator, HybridEvaluator, StateEvaluator


class MockNeuralEvaluator(StateEvaluator):
    def evaluate(self, state: BattleState) -> float:
        return 0.10


def test_supreme_overlord_scaling():
    """Kingambit's Kowtow Cleave damage must scale 10% per fainted ally up to 50%."""
    gambit = build_meta_pokemon("Kingambit", ["Kowtow Cleave"], "Black Glasses", PokemonType.FLYING, "physical", ability="Supreme Overlord")
    defender = build_meta_pokemon("Great Tusk", ["Close Combat"], "Leftovers", PokemonType.ICE, "bulky", ability="Protosynthesis")
    move = gambit.moves[0]

    rolls_0 = calculate_damage_rolls(gambit, defender, move, fallen_allies=0)
    rolls_1 = calculate_damage_rolls(gambit, defender, move, fallen_allies=1)
    rolls_3 = calculate_damage_rolls(gambit, defender, move, fallen_allies=3)
    rolls_5 = calculate_damage_rolls(gambit, defender, move, fallen_allies=5)

    med_0 = rolls_0[8]
    med_1 = rolls_1[8]
    med_3 = rolls_3[8]
    med_5 = rolls_5[8]

    # Damage with 5 fallen allies should be ~1.5x damage with 0 fallen
    ratio_5_to_0 = med_5 / med_0
    assert 1.45 <= ratio_5_to_0 <= 1.55, f"Supreme Overlord 5-fallen ratio expected ~1.50, got {ratio_5_to_0:.3f}"
    assert med_1 > med_0, "1 fallen ally must deal more damage than 0 fallen"
    assert med_5 > med_3, "5 fallen allies must deal more damage than 3 fallen"


def test_booster_energy_protosynthesis():
    """Great Tusk's highest stat (Atk) must receive a 30% boost from Booster Energy."""
    tusk_no_item = build_meta_pokemon("Great Tusk", ["Close Combat"], "Leftovers", PokemonType.ICE, "fast_phys", ability="Protosynthesis")
    tusk_booster = build_meta_pokemon("Great Tusk", ["Close Combat"], "Booster Energy", PokemonType.ICE, "fast_phys", ability="Protosynthesis")

    base_atk = tusk_no_item.effective_stat("atk")
    boosted_atk = tusk_booster.effective_stat("atk")

    ratio = boosted_atk / base_atk
    assert 1.28 <= ratio <= 1.32, f"Booster Energy expected 1.3x boost, got {ratio:.3f}"


def test_ability_immunities():
    """Verify Gen 9 ability immunities return 0 damage."""
    attacker = build_meta_pokemon("Great Tusk", ["Close Combat", "Headlong Rush"], "Leftovers", PokemonType.ICE, "fast_phys")
    water_move = Move.create("Surging Strikes", PokemonType.WATER, MoveCategory.PHYSICAL, base_power=25)
    fire_move = Move.create("Flamethrower", PokemonType.FIRE, MoveCategory.SPECIAL, base_power=90)
    ground_move = Move.create("Earthquake", PokemonType.GROUND, MoveCategory.PHYSICAL, base_power=100)
    electric_move = Move.create("Thunderbolt", PokemonType.ELECTRIC, MoveCategory.SPECIAL, base_power=90)

    # 1. Water Absorb (Ogerpon-Wellspring)
    ogerpon = build_meta_pokemon("Ogerpon-Wellspring", ["Ivy Cudgel"], "Wellspring Mask", PokemonType.WATER, ability="Water Absorb")
    rolls_water = calculate_damage_rolls(attacker, ogerpon, water_move)
    assert all(r == 0 for r in rolls_water), "Water move into Water Absorb must deal 0 damage"

    # 2. Flash Fire (Heatran)
    heatran = build_meta_pokemon("Heatran", ["Magma Storm"], "Leftovers", PokemonType.GRASS, ability="Flash Fire")
    rolls_fire = calculate_damage_rolls(attacker, heatran, fire_move)
    assert all(r == 0 for r in rolls_fire), "Fire move into Flash Fire must deal 0 damage"

    # 3. Levitate (Rotom-Wash)
    rotom = Pokemon(species="Rotom-Wash", ability="Levitate", types=(PokemonType.ELECTRIC, PokemonType.WATER))
    rolls_ground = calculate_damage_rolls(attacker, rotom, ground_move)
    assert all(r == 0 for r in rolls_ground), "Ground move into Levitate must deal 0 damage"

    # 4. Volt Absorb (Thundurus-Therian)
    thundurus = Pokemon(species="Thundurus-Therian", ability="Volt Absorb", types=(PokemonType.ELECTRIC, PokemonType.FLYING))
    rolls_elec = calculate_damage_rolls(attacker, thundurus, electric_move)
    assert all(r == 0 for r in rolls_elec), "Electric move into Volt Absorb must deal 0 damage"


def test_ruination_and_fixed_damage():
    """Ruination must cut target HP by exactly half (minimum 1), Night Shade deals level."""
    tinglu = build_meta_pokemon("Ting-Lu", ["Ruination"], "Leftovers", PokemonType.POISON, "bulky", ability="Vessel of Ruin")
    ruination = tinglu.moves[0]

    target_300 = Pokemon(species="Target", current_hp=300, max_hp=300)
    rolls_300 = calculate_damage_rolls(tinglu, target_300, ruination)
    assert all(r == 150 for r in rolls_300), f"Expected 150 damage on 300 HP target, got {rolls_300[0]}"

    target_17 = Pokemon(species="Target", current_hp=17, max_hp=100)
    rolls_17 = calculate_damage_rolls(tinglu, target_17, ruination)
    assert all(r == 8 for r in rolls_17), f"Expected 8 damage on 17 HP target, got {rolls_17[0]}"

    nightshade = Move.create("Night Shade", PokemonType.GHOST, MoveCategory.SPECIAL, base_power=1)
    rolls_ns = calculate_damage_rolls(tinglu, target_300, nightshade)
    assert rolls_ns == [0]*16  # Normal is immune to Ghost, including fixed damage.
    target_300.types = (PokemonType.WATER,None)
    rolls_ns = calculate_damage_rolls(tinglu,target_300,nightshade)
    assert all(r == tinglu.level for r in rolls_ns), f"Expected level {tinglu.level} damage, got {rolls_ns[0]}"


def test_no_double_burn():
    """Burned physical attacker must deal ~50% damage of healthy attacker, NOT 25%."""
    tusk_healthy = build_meta_pokemon("Great Tusk", ["Close Combat"], "Leftovers", PokemonType.ICE, "fast_phys")
    tusk_burned = build_meta_pokemon("Great Tusk", ["Close Combat"], "Leftovers", PokemonType.ICE, "fast_phys")
    tusk_burned.status = StatusCondition.BURN
    defender = build_meta_pokemon("Ting-Lu", ["Earthquake"], "Leftovers", PokemonType.POISON, "bulky")

    cc = tusk_healthy.moves[0]
    rolls_healthy = calculate_damage_rolls(tusk_healthy, defender, cc)
    rolls_burned = calculate_damage_rolls(tusk_burned, defender, cc)

    ratio = rolls_burned[8] / rolls_healthy[8]
    assert 0.47 <= ratio <= 0.53, f"Burned physical move ratio expected ~0.50, got {ratio:.3f}"


def test_action_to_logit_index_mapping():
    """All 14 discrete actions (4 moves, 4 tera moves, 6 switches) must map to distinct logits 0..13."""
    seen_indices = set()
    for slot in range(1, 5):
        m_act = MoveAction(move_id="tackle", move_slot=slot, is_tera=False)
        tera_act = MoveAction(move_id="tackle", move_slot=slot, is_tera=True)
        idx_m = action_to_logit_index(m_act)
        idx_t = action_to_logit_index(tera_act)
        assert idx_m == slot - 1, f"Move slot {slot} should be {slot - 1}"
        assert idx_t == slot + 3, f"Tera move slot {slot} should be {slot + 3}"
        seen_indices.add(idx_m)
        seen_indices.add(idx_t)

    for target in range(1, 7):
        sw_act = SwitchAction(target_slot=target, species="TestMon")
        idx_s = action_to_logit_index(sw_act)
        assert idx_s == 7 + target, f"Switch to party slot {target} should map to {7 + target}, got {idx_s}"
        seen_indices.add(idx_s)

    assert seen_indices == set(range(14)), f"All 14 logits must be covered! Got: {seen_indices}"


def test_stealth_rock_and_rapid_spin_simulation():
    """Stealth Rock adds hazard to opponent side, Rapid Spin clears hazards."""
    team1 = get_meta_team_balance()
    team2 = get_meta_team_balance()
    state = BattleState(p1=BattleSide(pokemon=team1, active_index=5), p2=BattleSide(pokemon=team2, active_index=0))

    # Turn 1: Ting-Lu clicks Stealth Rock (move_slot 3)
    sr_act = MoveAction(move_id="stealthrock", move_slot=3)
    opp_act = MoveAction(move_id="closecombat", move_slot=1)
    s_after = simulate_turn_transition(state, sr_act, opp_act)

    assert Hazard.STEALTH_ROCK in s_after.p2.hazards, "Stealth Rock must be on P2 hazards after turn"

    # Turn 2: P2 Great Tusk uses Rapid Spin (move_slot 4)
    rs_act = MoveAction(move_id="rapidspin", move_slot=4)
    tinglu_act = MoveAction(move_id="earthquake", move_slot=1)
    s_after_spin = simulate_turn_transition(s_after, tinglu_act, rs_act)
    assert len(s_after_spin.p2.hazards) == 0, "Rapid Spin must clear hazards from user side"


def test_end_of_turn_leftovers_and_burn():
    """End-of-turn phase applies Leftovers heal and Burn chip."""
    p1_mon = Pokemon(species="Ting-Lu", current_hp=200, max_hp=300, item="Leftovers")
    p2_mon = Pokemon(species="Great Tusk", current_hp=200, max_hp=300, status=StatusCondition.BURN)

    state = BattleState(
        p1=BattleSide(pokemon=[p1_mon], active_index=0),
        p2=BattleSide(pokemon=[p2_mon], active_index=0)
    )

    a1 = MoveAction(move_id="spikyshield", move_slot=1)
    a2 = MoveAction(move_id="spikyshield", move_slot=1)
    s_next = simulate_turn_transition(state, a1, a2)

    # Leftovers: +18 HP (300 // 16 = 18)
    assert s_next.p1.active_pokemon.current_hp == 218, f"Expected 218 HP, got {s_next.p1.active_pokemon.current_hp}"
    # Burn: -18 HP (300 // 16 = 18)
    assert s_next.p2.active_pokemon.current_hp == 182, f"Expected 182 HP, got {s_next.p2.active_pokemon.current_hp}"


def test_principled_ko_decision_without_hardcoding():
    """When opponent Kingambit is at 50 HP (in KO range), bot attacks without any hardcoded penalty."""
    team1 = get_meta_team_balance()
    team2 = get_meta_team_balance()
    s1 = BattleSide(pokemon=[p.clone() for p in team1], active_index=2)
    s2 = BattleSide(pokemon=[p.clone() for p in team2], active_index=2)
    for i in [0, 1, 3, 4, 5]:
        s1.pokemon[i].current_hp = 0
        s2.pokemon[i].current_hp = 0

    s2.pokemon[2].current_hp = 50  # KO range

    state = BattleState(p1=s1, p2=s2)
    evaluator = HeuristicEvaluator()
    resolver = SubgameResolver(evaluator=evaluator)

    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)
    assert action.action_type == ActionType.MOVE
    assert action.move_id in ("kowtowcleave", "ironhead", "suckerpunch"), f"Must choose damaging move in KO range, chose: {action.move_id}"
    assert action.move_id != "swordsdance", "Must NOT click Swords Dance when opponent is in KO range!"


def test_hybrid_evaluator_blend():
    """HybridEvaluator blends neural and heuristic evaluations with specified weight."""
    mock_nn = MockNeuralEvaluator()
    heur = HeuristicEvaluator()
    hybrid = HybridEvaluator(neural_evaluator=mock_nn, heuristic_evaluator=heur, weight_neural=0.60)

    team1 = get_meta_team_balance()
    team2 = get_meta_team_balance()
    state = BattleState(p1=BattleSide(pokemon=team1, active_index=0), p2=BattleSide(pokemon=team2, active_index=0))

    v_nn = mock_nn.evaluate(state)
    v_heur = heur.evaluate(state)
    v_hyb = hybrid.evaluate(state)

    expected = 0.60 * v_nn + 0.40 * v_heur
    assert abs(v_hyb - expected) < 1e-4, f"Hybrid evaluation {v_hyb} != expected {expected}"


def test_consecutive_protect_decay():
    """Turn 1 of Protect takes 0 damage, Turn 2 with streak=1 takes expected damage, avoiding spam."""
    ogerpon = build_meta_pokemon("Ogerpon-Wellspring", ["Spiky Shield", "Ivy Cudgel"], "Wellspring Mask", PokemonType.WATER)
    dragapult = build_meta_pokemon("Dragapult", ["Flamethrower"], "Choice Specs", PokemonType.GHOST)

    state1 = BattleState(
        p1=BattleSide(pokemon=[ogerpon.clone()], active_index=0),
        p2=BattleSide(pokemon=[dragapult.clone()], active_index=0)
    )
    # Streak 0: 100% success -> 0 damage
    a1 = MoveAction(move_id="spikyshield", move_slot=1)
    a2 = MoveAction(move_id="flamethrower", move_slot=1)
    s_turn1 = simulate_turn_transition(state1, a1, a2)
    assert s_turn1.p1.active_pokemon.current_hp == ogerpon.max_hp
    assert s_turn1.p1.active_pokemon.protect_streak == 1

    # Streak 1: Success rate 1/3 -> takes 2/3 damage
    s_turn2 = simulate_turn_transition(s_turn1, a1, a2)
    assert s_turn2.p1.active_pokemon.current_hp < ogerpon.max_hp

    # SubgameResolver should NOT select Spiky Shield when streak=1 against offensive threat
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    act, strat, _, _ = resolver.resolve_turn(s_turn1, depth=1, sample=False)
    assert act.move_id != "spikyshield", "Bot must not spam Spiky Shield when protect streak > 0"


def test_insolvent_recovery_rejection():
    """Gholdengo must attack rather than clicking Recover when slower opponent deals >50% damage."""
    gholdengo = build_meta_pokemon("Gholdengo", ["Recover", "Shadow Ball", "Make It Rain"], None, PokemonType.STEEL, "bulky")
    gholdengo.current_hp = 250
    ogerpon = build_meta_pokemon("Ogerpon-Wellspring", ["Ivy Cudgel"], "Wellspring Mask", PokemonType.WATER, "fast_phys")

    state = BattleState(
        p1=BattleSide(pokemon=[gholdengo], active_index=0),
        p2=BattleSide(pokemon=[ogerpon], active_index=0)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    act, strat, actions, _ = resolver.resolve_turn(state, depth=1, sample=False)

    assert act.action_type == ActionType.MOVE
    assert act.move_id != "recover", "Gholdengo must not click Recover in an insolvent recovery trap!"


def test_knock_off_mechanics():
    """Knock Off deals 1.5x damage against an item-holding defender and removes the item."""
    attacker = build_meta_pokemon("Great Tusk", ["Knock Off"], "Booster Energy", PokemonType.ICE)
    defender = build_meta_pokemon("Gholdengo", ["Shadow Ball"], "Leftovers", PokemonType.STEEL)

    state = BattleState(
        p1=BattleSide(pokemon=[attacker], active_index=0),
        p2=BattleSide(pokemon=[defender], active_index=0)
    )
    a1 = MoveAction(move_id="knockoff", move_slot=1)
    a2 = MoveAction(move_id="shadowball", move_slot=1)

    assert defender.item == "Leftovers"
    s_next = simulate_turn_transition(state, a1, a2)
    assert s_next.p2.active_pokemon.item is None, "Knock Off must remove the defender's item!"


def test_recoil_moves_mechanics():
    """Recoil moves (Brave Bird, Flare Blitz, Wave Crash) deal 33% recoil to the attacker."""
    bird = Pokemon(
        species="Corviknight",
        types=(PokemonType.FLYING, PokemonType.STEEL),
        raw_stats={"hp": 398, "atk": 250, "def": 300, "spa": 100, "spd": 250, "spe": 180},
        moves=[Move(id="bravebird", name="Brave Bird", move_type=PokemonType.FLYING, category=MoveCategory.PHYSICAL, base_power=120)]
    )
    dummy = Pokemon(
        species="Mew",
        types=(PokemonType.PSYCHIC, None),
        raw_stats={"hp": 400, "atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 100},
        moves=[Move(id="recover", name="Recover", move_type=PokemonType.NORMAL, category=MoveCategory.STATUS, base_power=0)]
    )
    state = BattleState(
        p1=BattleSide(pokemon=[bird], active_index=0),
        p2=BattleSide(pokemon=[dummy], active_index=0)
    )
    s_next = simulate_turn_transition(state, MoveAction("bravebird", 1), MoveAction("recover", 1))
    assert s_next.p1.active_pokemon.current_hp < bird.max_hp, "Attacker must take recoil damage from Brave Bird!"


def test_rocky_helmet_contact_recoil():
    """Attacking a Rocky Helmet holder with a contact move inflicts 1/6 max HP recoil to the attacker."""
    attacker = Pokemon(
        species="Great Tusk",
        types=(PokemonType.GROUND, PokemonType.FIGHTING),
        raw_stats={"hp": 371, "atk": 361, "def": 301, "spa": 127, "spd": 142, "spe": 213},
        moves=[Move(id="closecombat", name="Close Combat", move_type=PokemonType.FIGHTING, category=MoveCategory.PHYSICAL, base_power=120)]
    )
    defender = Pokemon(
        species="Corviknight",
        types=(PokemonType.FLYING, PokemonType.STEEL),
        raw_stats={"hp": 398, "atk": 200, "def": 300, "spa": 100, "spd": 250, "spe": 100},
        item="Rocky Helmet",
        moves=[Move(id="roost", name="Roost", move_type=PokemonType.FLYING, category=MoveCategory.STATUS, base_power=0)]
    )
    state = BattleState(
        p1=BattleSide(pokemon=[attacker], active_index=0),
        p2=BattleSide(pokemon=[defender], active_index=0)
    )
    s_next = simulate_turn_transition(state, MoveAction("closecombat", 1), MoveAction("roost", 1))
    expected_recoil = attacker.max_hp // 6
    dmg_taken = attacker.max_hp - s_next.p1.active_pokemon.current_hp
    assert dmg_taken >= expected_recoil, f"Rocky Helmet should deal >= {expected_recoil} chip, dealt {dmg_taken}"


def test_regenerator_healing_on_switch():
    """Regenerator heals 33% max HP when the user switches out."""
    slowking = Pokemon(
        species="Slowking-Galar",
        types=(PokemonType.POISON, PokemonType.PSYCHIC),
        raw_stats={"hp": 394, "atk": 150, "def": 200, "spa": 250, "spd": 300, "spe": 100},
        current_hp=200,
        max_hp=394,
        ability="Regenerator",
        moves=[Move(id="chillyreception", name="Chilly Reception", move_type=PokemonType.ICE, category=MoveCategory.STATUS, base_power=0)]
    )
    bench_mon = Pokemon(
        species="Great Tusk",
        types=(PokemonType.GROUND, PokemonType.FIGHTING),
        raw_stats={"hp": 371, "atk": 361, "def": 301, "spa": 127, "spd": 142, "spe": 213},
        moves=[]
    )
    state = BattleState(
        p1=BattleSide(pokemon=[slowking, bench_mon], active_index=0),
        p2=BattleSide(pokemon=[bench_mon.clone()], active_index=0)
    )
    s_next = simulate_turn_transition(state, SwitchAction(target_slot=2, species="Great Tusk"), MoveAction("closecombat", 1))
    switched_slowking = s_next.p1.pokemon[0]
    expected_heal = slowking.max_hp // 3
    assert switched_slowking.current_hp == 200 + expected_heal, "Regenerator must heal 33% max HP on switch out!"


def test_protect_blocks_pivot_switch():
    """When Protect blocks U-turn, the attacker does NOT switch out."""
    dragapult = Pokemon(
        species="Dragapult",
        types=(PokemonType.DRAGON, PokemonType.GHOST),
        raw_stats={"hp": 317, "atk": 240, "def": 186, "spa": 299, "spd": 186, "spe": 383},
        moves=[Move(id="uturn", name="U-turn", move_type=PokemonType.BUG, category=MoveCategory.PHYSICAL, base_power=70)]
    )
    bench_mon = Pokemon(
        species="Ting-Lu",
        types=(PokemonType.DARK, PokemonType.GROUND),
        raw_stats={"hp": 514, "atk": 256, "def": 286, "spa": 135, "spd": 228, "spe": 126},
        moves=[]
    )
    ogerpon = Pokemon(
        species="Ogerpon-Wellspring",
        types=(PokemonType.GRASS, PokemonType.WATER),
        raw_stats={"hp": 301, "atk": 279, "def": 204, "spa": 140, "spd": 228, "spe": 319},
        moves=[Move(id="spikyshield", name="Spiky Shield", move_type=PokemonType.GRASS, category=MoveCategory.STATUS, base_power=0, priority=4)]
    )
    state = BattleState(
        p1=BattleSide(pokemon=[dragapult, bench_mon], active_index=0),
        p2=BattleSide(pokemon=[ogerpon], active_index=0)
    )
    s_next = simulate_turn_transition(state, MoveAction("uturn", 1), MoveAction("spikyshield", 1))
    assert s_next.p1.active_index == 0, "U-turn must NOT switch out if blocked by Protect / Spiky Shield!"


def test_unaware_ignores_stat_boosts():
    """Unaware ignores the opponent's offensive and defensive stat stage modifiers."""
    clodsire = Pokemon(
        species="Clodsire",
        types=(PokemonType.POISON, PokemonType.GROUND),
        raw_stats={"hp": 400, "atk": 200, "def": 200, "spa": 100, "spd": 300, "spe": 100},
        ability="Unaware"
    )
    tusk_unboosted = Pokemon(
        species="Great Tusk",
        types=(PokemonType.GROUND, PokemonType.FIGHTING),
        raw_stats={"hp": 371, "atk": 300, "def": 200, "spa": 100, "spd": 200, "spe": 200}
    )
    tusk_boosted = Pokemon(
        species="Great Tusk",
        types=(PokemonType.GROUND, PokemonType.FIGHTING),
        raw_stats={"hp": 371, "atk": 300, "def": 200, "spa": 100, "spd": 200, "spe": 200},
        boosts={"atk": 2}
    )
    eq = Move(id="earthquake", name="Earthquake", move_type=PokemonType.GROUND, category=MoveCategory.PHYSICAL, base_power=100)

    rolls_unboosted = calculate_damage_rolls(tusk_unboosted, clodsire, eq)
    rolls_boosted = calculate_damage_rolls(tusk_boosted, clodsire, eq)

    assert rolls_unboosted == rolls_boosted, "Unaware defender must take identical damage regardless of attacker's +2 Atk!"


def test_technician_boosts_low_power_moves():
    """Technician boosts moves with base power <= 60 by 1.5x."""
    scizor = Pokemon(
        species="Scizor",
        types=(PokemonType.BUG, PokemonType.STEEL),
        raw_stats={"hp": 300, "atk": 350, "def": 200, "spa": 100, "spd": 200, "spe": 150},
        ability="Technician"
    )
    scizor_no_tech = Pokemon(
        species="Scizor",
        types=(PokemonType.BUG, PokemonType.STEEL),
        raw_stats={"hp": 300, "atk": 350, "def": 200, "spa": 100, "spd": 200, "spe": 150},
        ability="Swarm"
    )
    dummy = Pokemon(
        species="Mew",
        types=(PokemonType.PSYCHIC, None),
        raw_stats={"hp": 300, "atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}
    )
    bp = Move(id="bulletpunch", name="Bullet Punch", move_type=PokemonType.STEEL, category=MoveCategory.PHYSICAL, base_power=40)

    rolls_tech = calculate_damage_rolls(scizor, dummy, bp)
    rolls_norm = calculate_damage_rolls(scizor_no_tech, dummy, bp)

    assert rolls_tech[8] > rolls_norm[8]
    assert 1.45 <= (rolls_tech[8] / rolls_norm[8]) <= 1.55, "Technician must multiply damage by 1.5x!"


def test_poison_heal_gliscor():
    """Gliscor with Poison Heal heals 1/8 max HP when poisoned instead of taking damage."""
    gliscor = Pokemon(
        species="Gliscor",
        types=(PokemonType.GROUND, PokemonType.FLYING),
        raw_stats={"hp": 354, "atk": 200, "def": 280, "spa": 100, "spd": 200, "spe": 250},
        current_hp=200,
        max_hp=354,
        status=StatusCondition.TOXIC,
        ability="Poison Heal",
        moves=[Move(id="earthquake", name="Earthquake", move_type=PokemonType.GROUND, category=MoveCategory.PHYSICAL, base_power=100)]
    )
    dummy = Pokemon(
        species="Mew",
        types=(PokemonType.PSYCHIC, None),
        raw_stats={"hp": 300, "atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200},
        moves=[Move(id="protect", name="Protect", move_type=PokemonType.NORMAL, category=MoveCategory.STATUS, base_power=0, priority=4)]
    )
    state = BattleState(
        p1=BattleSide(pokemon=[gliscor], active_index=0),
        p2=BattleSide(pokemon=[dummy], active_index=0)
    )
    s_next = simulate_turn_transition(state, MoveAction("earthquake", 1), MoveAction("protect", 1))
    expected_heal = gliscor.max_hp // 8
    assert s_next.p1.active_pokemon.current_hp == 200 + expected_heal, "Poison Heal must recover 1/8 max HP!"


def test_hazard_attacks_ceaseless_edge_and_stone_axe():
    """Ceaseless Edge sets Spikes and Stone Axe sets Stealth Rock upon dealing damage."""
    samurott = Pokemon(
        species="Samurott-Hisui",
        types=(PokemonType.WATER, PokemonType.DARK),
        raw_stats={"hp": 321, "atk": 320, "def": 190, "spa": 150, "spd": 160, "spe": 260},
        moves=[Move(id="ceaselessedge", name="Ceaseless Edge", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=65)]
    )
    kleavor = Pokemon(
        species="Kleavor",
        types=(PokemonType.BUG, PokemonType.ROCK),
        raw_stats={"hp": 300, "atk": 360, "def": 220, "spa": 100, "spd": 180, "spe": 250},
        moves=[Move(id="stoneaxe", name="Stone Axe", move_type=PokemonType.ROCK, category=MoveCategory.PHYSICAL, base_power=65)]
    )
    dummy = Pokemon(
        species="Mew",
        types=(PokemonType.PSYCHIC, None),
        raw_stats={"hp": 300, "atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200},
        moves=[Move(id="recover", name="Recover", move_type=PokemonType.NORMAL, category=MoveCategory.STATUS, base_power=0)]
    )
    state1 = BattleState(p1=BattleSide(pokemon=[samurott], active_index=0), p2=BattleSide(pokemon=[dummy.clone()], active_index=0))
    s_next1 = simulate_turn_transition(state1, MoveAction("ceaselessedge", 1), MoveAction("recover", 1))
    assert Hazard.SPIKES_1 in s_next1.p2.hazards, "Ceaseless Edge must set Spikes on foe side!"

    state2 = BattleState(p1=BattleSide(pokemon=[kleavor], active_index=0), p2=BattleSide(pokemon=[dummy.clone()], active_index=0))
    s_next2 = simulate_turn_transition(state2, MoveAction("stoneaxe", 1), MoveAction("recover", 1))
    assert Hazard.STEALTH_ROCK in s_next2.p2.hazards, "Stone Axe must set Stealth Rock on foe side!"


def test_magic_bounce_reflection():
    """Magic Bounce reflects Stealth Rock and status moves back to the attacker."""
    attacker = Pokemon(
        species="Ting-Lu",
        types=(PokemonType.DARK, PokemonType.GROUND),
        raw_stats={"hp": 514, "atk": 256, "def": 286, "spa": 135, "spd": 228, "spe": 126},
        moves=[Move(id="stealthrock", name="Stealth Rock", move_type=PokemonType.ROCK, category=MoveCategory.STATUS, base_power=0)]
    )
    hatterene = Pokemon(
        species="Hatterene",
        types=(PokemonType.PSYCHIC, PokemonType.FAIRY),
        raw_stats={"hp": 318, "atk": 150, "def": 220, "spa": 350, "spd": 240, "spe": 90},
        ability="Magic Bounce",
        moves=[Move(id="dazzlinggleam", name="Dazzling Gleam", move_type=PokemonType.FAIRY, category=MoveCategory.SPECIAL, base_power=80)]
    )
    state = BattleState(p1=BattleSide(pokemon=[attacker], active_index=0), p2=BattleSide(pokemon=[hatterene], active_index=0))
    s_next = simulate_turn_transition(state, MoveAction("stealthrock", 1), MoveAction("dazzlinggleam", 1))
    assert Hazard.STEALTH_ROCK in s_next.p1.hazards, "Magic Bounce must bounce Stealth Rock back onto the user's side!"
    assert Hazard.STEALTH_ROCK not in s_next.p2.hazards


def test_gholdengo_lethal_satiation_turn38():
    """Gholdengo at +2 SpA against 13% Kingambit must attack to close the match instead of redundant setup."""
    gholdengo = Pokemon(
        species="Gholdengo", types=(PokemonType.STEEL, PokemonType.GHOST),
        raw_stats={"hp": 315, "atk": 140, "def": 226, "spa": 365, "spd": 218, "spe": 204},
        current_hp=315, max_hp=315, status=StatusCondition.NONE,
        moves=[
            Move(id="makeitrain", name="Make It Rain", move_type=PokemonType.STEEL, category=MoveCategory.SPECIAL, base_power=120),
            Move(id="shadowball", name="Shadow Ball", move_type=PokemonType.GHOST, category=MoveCategory.SPECIAL, base_power=80),
            Move(id="nastyplot", name="Nasty Plot", move_type=PokemonType.DARK, category=MoveCategory.STATUS, base_power=0),
            Move(id="recover", name="Recover", move_type=PokemonType.NORMAL, category=MoveCategory.STATUS, base_power=0),
        ],
        boosts={"spa": 2}, item="airballoon", ability="Good as Gold"
    )
    kingambit = Pokemon(
        species="Kingambit", types=(PokemonType.DARK, PokemonType.STEEL),
        raw_stats={"hp": 341, "atk": 405, "def": 276, "spa": 140, "spd": 206, "spe": 136},
        current_hp=39, max_hp=300, status=StatusCondition.NONE,
        moves=[
            Move(id="kowtowcleave", name="Kowtow Cleave", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=85),
            Move(id="suckerpunch", name="Sucker Punch", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=70, priority=1),
            Move(id="ironhead", name="Iron Head", move_type=PokemonType.STEEL, category=MoveCategory.PHYSICAL, base_power=80),
            Move(id="swordsdance", name="Swords Dance", move_type=PokemonType.NORMAL, category=MoveCategory.STATUS, base_power=0),
        ],
        boosts={"atk": 6}, item="leftovers", ability="Supreme Overlord"
    )
    state = BattleState(
        p1=BattleSide(pokemon=[gholdengo], active_index=0),
        p2=BattleSide(pokemon=[kingambit], active_index=0)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    assert action.action_type == ActionType.MOVE
    assert action.move_id in ("makeitrain", "shadowball"), f"Gholdengo must execute lethal attack, chose: {action.move_id}"
    np_strat = next((p for a, p in zip(actions, strat) if getattr(a, "move_id", "") == "nastyplot"), 0.0)
    rec_strat = next((p for a, p in zip(actions, strat) if getattr(a, "move_id", "") == "recover"), 0.0)
    assert np_strat < 0.05, f"Nasty Plot strategy must be < 5% when target in lethal range, got {np_strat}"
    assert rec_strat < 0.05, f"Recover strategy must be < 5% when at full HP, got {rec_strat}"


def test_ogerpon_consecutive_protect_penalty():
    """Ogerpon with protect streak >= 2 must choose attacks 100%, and streak 1 must mix attacks."""
    ogerpon = Pokemon(
        species="Ogerpon-Wellspring", types=(PokemonType.GRASS, PokemonType.WATER),
        raw_stats={"hp": 301, "atk": 279, "def": 204, "spa": 140, "spd": 228, "spe": 319},
        current_hp=18, max_hp=301, status=StatusCondition.NONE,
        moves=[
            Move(id="ivycudgel", name="Ivy Cudgel", move_type=PokemonType.WATER, category=MoveCategory.PHYSICAL, base_power=100),
            Move(id="spikyshield", name="Spiky Shield", move_type=PokemonType.GRASS, category=MoveCategory.STATUS, base_power=0, priority=4),
        ],
        item="wellspringmask", ability="Water Absorb",
        protect_streak=2
    )
    kingambit = Pokemon(
        species="Kingambit", types=(PokemonType.DARK, PokemonType.STEEL),
        raw_stats={"hp": 341, "atk": 405, "def": 276, "spa": 140, "spd": 206, "spe": 136},
        current_hp=114, max_hp=300, status=StatusCondition.NONE,
        moves=[
            Move(id="kowtowcleave", name="Kowtow Cleave", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=85),
            Move(id="suckerpunch", name="Sucker Punch", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=70, priority=1),
        ],
        boosts={"atk": 2}, item="leftovers", ability="Supreme Overlord"
    )
    state = BattleState(
        p1=BattleSide(pokemon=[ogerpon], active_index=0),
        p2=BattleSide(pokemon=[kingambit], active_index=0)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    assert action.action_type == ActionType.MOVE
    assert action.move_id == "ivycudgel", f"Ogerpon at streak 2 must attack rather than loop protect, chose: {action.move_id}"


def test_switch_in_faint_penalty_rejection():
    """Turn 2 Regression: Great Tusk at 46% must NOT switch in Dragapult into lethal Ice Spinner."""
    from glaubermon.data.meta_teams import get_meta_team_balance
    team1 = get_meta_team_balance()
    team2 = get_meta_team_balance()
    team1[0].current_hp = int(team1[0].max_hp * 0.46)
    team2[0].current_hp = int(team2[0].max_hp * 0.46)
    team1[0].boosts["atk"] = 1
    team2[0].boosts["atk"] = 1
    team1[0].item = None
    team2[0].item = None

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=0),
        p2=BattleSide(pokemon=team2, active_index=0)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    assert action.action_type == ActionType.MOVE, f"Great Tusk must stay in and attack, chose: {action}"
    drag_prob = next((p for a, p in zip(actions, strat) if getattr(a, "species", "") == "Dragapult"), 0.0)
    assert drag_prob < 0.01, f"Dragapult suicide switch probability must be < 1%, got {drag_prob * 100:.1f}%"


def test_whirlwind_purposeless_penalty():
    """Turn 4 Regression: Ting-Lu must NOT click Whirlwind when opponent has no boosts and no hazards."""
    from glaubermon.data.meta_teams import get_meta_team_balance
    team1 = get_meta_team_balance()
    team2 = get_meta_team_balance()
    team1[5].current_hp = int(team1[5].max_hp * 0.70)
    team2[0].current_hp = int(team2[0].max_hp * 0.935)

    state = BattleState(
        p1=BattleSide(pokemon=team1, active_index=5),
        p2=BattleSide(pokemon=team2, active_index=0)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    whirl_prob = next((p for a, p in zip(actions, strat) if getattr(a, "move_id", "") == "whirlwind"), 0.0)
    assert whirl_prob < 0.01, f"Whirlwind on unboosted, unhazarded target must be < 1%, got {whirl_prob * 100:.1f}%"


def test_sucker_punch_anti_stall_escalation():
    """Kingambit at 1 HP against Ogerpon Spiky Shield/Setup must not loop Sucker Punch when streak >= 1."""
    king = Pokemon(
        species="Kingambit", types=(PokemonType.DARK, PokemonType.STEEL),
        raw_stats={"hp": 341, "atk": 405, "def": 276, "spa": 140, "spd": 206, "spe": 136},
        current_hp=1, max_hp=341, status=StatusCondition.NONE,
        moves=[
            Move(id="kowtowcleave", name="Kowtow Cleave", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=85),
            Move(id="suckerpunch", name="Sucker Punch", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=70, priority=1),
            Move(id="ironhead", name="Iron Head", move_type=PokemonType.STEEL, category=MoveCategory.PHYSICAL, base_power=80),
            Move(id="swordsdance", name="Swords Dance", move_type=PokemonType.NORMAL, category=MoveCategory.STATUS, base_power=0),
        ],
        ability="Supreme Overlord"
    )
    oger = Pokemon(
        species="Ogerpon-Wellspring", types=(PokemonType.GRASS, PokemonType.WATER),
        raw_stats={"hp": 301, "atk": 339, "def": 204, "spa": 140, "spd": 228, "spe": 319},
        current_hp=301, max_hp=301, status=StatusCondition.NONE,
        moves=[
            Move(id="ivycudgel", name="Ivy Cudgel", move_type=PokemonType.WATER, category=MoveCategory.PHYSICAL, base_power=100),
            Move(id="spikyshield", name="Spiky Shield", move_type=PokemonType.GRASS, category=MoveCategory.STATUS, base_power=0, priority=4),
            Move(id="swordsdance", name="Swords Dance", move_type=PokemonType.NORMAL, category=MoveCategory.STATUS, base_power=0),
        ]
    )
    state = BattleState(
        p1=BattleSide(pokemon=[king], active_index=0),
        p2=BattleSide(pokemon=[oger], active_index=0)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False, p1_sucker_streak=1)
    sucker_prob = next((p for a, p in zip(actions, strat) if getattr(a, "move_id", "") == "suckerpunch"), 0.0)
    assert sucker_prob < 0.05, f"Sucker punch probability on streak >= 1 must be < 5%, got {sucker_prob * 100:.1f}%"
    assert action.move_id in ("kowtowcleave", "swordsdance", "ironhead")


def test_booster_energy_voluntary_switch_rejection():
    """Active Great Tusk with Booster Energy Attack boost must attack and NOT throw away boost on switch."""
    from glaubermon.data.meta_teams import get_meta_team_balance
    p1_mons = get_meta_team_balance()
    p2_mons = get_meta_team_balance()
    p1_tusk_idx = next(i for i, p in enumerate(p1_mons) if "greattusk" in p.species.lower().replace(" ", "").replace("-", ""))
    p2_oger_idx = next(i for i, p in enumerate(p2_mons) if "ogerpon" in p.species.lower().replace(" ", "").replace("-", ""))
    p1_tusk = p1_mons[p1_tusk_idx]
    p1_tusk.booster_stat = "atk"
    p1_tusk.item = None

    state = BattleState(
        p1=BattleSide(pokemon=p1_mons, active_index=p1_tusk_idx),
        p2=BattleSide(pokemon=p2_mons, active_index=p2_oger_idx)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)
    assert action.action_type == ActionType.MOVE, f"Booster Tusk must stay in and attack, chose switch: {action}"


def test_gholdengo_air_balloon_nasty_plot_vulnerability():
    """Gholdengo with Air Balloon against faster Great Tusk must not Nasty Plot into lethal Ground attack."""
    from glaubermon.data.meta_teams import get_meta_team_balance
    p1_mons = get_meta_team_balance()
    p2_mons = get_meta_team_balance()
    p1_ghold_idx = next(i for i, p in enumerate(p1_mons) if "gholdengo" in p.species.lower().replace(" ", "").replace("-", ""))
    p2_tusk_idx = next(i for i, p in enumerate(p2_mons) if "greattusk" in p.species.lower().replace(" ", "").replace("-", ""))

    state = BattleState(
        p1=BattleSide(pokemon=p1_mons, active_index=p1_ghold_idx),
        p2=BattleSide(pokemon=p2_mons, active_index=p2_tusk_idx)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=2, sample=False)
    assert getattr(action, "move_id", "") != "nastyplot", f"Gholdengo must not use Nasty Plot when Air Balloon pops to faster lethal threat"


def test_showdown_bot_evaluator_wiring():
    """Verify that ShowdownBot properly connects the requested evaluator to SubgameResolver."""
    from glaubermon.client.showdown_bot import ShowdownBot
    from glaubermon.search.evaluators import HybridEvaluator, HeuristicEvaluator, NeuralEvaluator

    # Default evaluator mode: Hybrid (blends neural network + heuristic)
    bot_hybrid = ShowdownBot(username="TestBot", evaluator="hybrid", checkpoint="checkpoints/glaubermon_rebel_latest.pt")
    assert isinstance(bot_hybrid.evaluator, HybridEvaluator), "Default evaluator must be HybridEvaluator"
    assert bot_hybrid.resolver.evaluator is bot_hybrid.evaluator
    assert hasattr(bot_hybrid.evaluator, "get_policy_prior"), "HybridEvaluator must provide get_policy_prior"

    # Pure heuristic mode
    bot_heur = ShowdownBot(username="TestBot", evaluator="heuristic")
    assert isinstance(bot_heur.evaluator, HeuristicEvaluator), "evaluator='heuristic' must set HeuristicEvaluator"
    assert bot_heur.resolver.evaluator is bot_heur.evaluator

    # Pure neural mode
    bot_neural = ShowdownBot(username="TestBot", evaluator="neural", checkpoint="checkpoints/glaubermon_rebel_latest.pt")
    assert isinstance(bot_neural.evaluator, NeuralEvaluator), "evaluator='neural' must set NeuralEvaluator"
    assert bot_neural.resolver.evaluator is bot_neural.evaluator
