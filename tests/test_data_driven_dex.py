"""Unit tests verifying authoritative ShowdownDex data loading and data-driven mechanics."""

import pytest
from glaubermon.core.types import PokemonType, MoveCategory, StatusCondition, Hazard
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.actions import MoveAction, SwitchAction
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.constants import clean_key
from glaubermon.data.showdown_dex import ShowdownDex
from glaubermon.search.subgame_resolver import simulate_turn_transition, apply_entry_hazards


def test_showdown_dex_canonical_attributes():
    """Verify that ShowdownDex loads authoritative data for competitive Gen 9 moves."""
    dex = ShowdownDex.get_instance()

    # Sucker Punch
    sp = dex.get_move("suckerpunch")
    assert sp.pp == 8
    assert sp.max_pp == 8
    assert sp.priority == 1
    assert sp.is_contact is True
    assert sp.category == MoveCategory.PHYSICAL
    assert sp.move_type == PokemonType.DARK

    # Spiky Shield
    ss = dex.get_move("spikyshield")
    assert ss.pp == 16
    assert ss.max_pp == 16
    assert ss.priority == 4
    assert ss.is_protect is True
    assert ss.category == MoveCategory.STATUS

    # Protect
    prot = dex.get_move("protect")
    assert prot.pp == 16
    assert prot.is_protect is True
    assert prot.priority == 4

    # Bitter Blade (DLC move: drain 50%, slicing, contact)
    bb = dex.get_move("bitterblade")
    assert bb.drain == (1, 2)
    assert bb.is_slicing is True
    assert bb.is_contact is True
    assert bb.base_power == 90

    # Horn Leech (drain 50%, contact)
    hl = dex.get_move("hornleech")
    assert hl.drain == (1, 2)
    assert hl.is_contact is True

    # Brave Bird (recoil 33%, contact)
    bbird = dex.get_move("bravebird")
    assert bbird.recoil == (33, 100)
    assert bbird.is_contact is True

    # Head Smash (recoil 50%, contact)
    hs = dex.get_move("headsmash")
    assert hs.recoil == (1, 2)
    assert hs.is_contact is True

    # Close Combat (self drops -1 Def, -1 SpD)
    cc = dex.get_move("closecombat")
    assert cc.self_boosts == {"def": -1, "spd": -1}
    assert cc.is_contact is True

    # Swords Dance (self +2 Atk)
    sd = dex.get_move("swordsdance")
    assert sd.self_boosts == {"atk": 2}

    # Dragon Dance (self +1 Atk, +1 Spe)
    dd = dex.get_move("dragondance")
    assert dd.self_boosts == {"atk": 1, "spe": 1}

    # Calm Mind (self +1 SpA, +1 SpD)
    cm = dex.get_move("calmmind")
    assert cm.self_boosts == {"spa": 1, "spd": 1}

    # Make It Rain (self -1 SpA)
    mir = dex.get_move("makeitrain")
    assert mir.self_boosts == {"spa": -1}

    # Draco Meteor (self -2 SpA)
    dm = dex.get_move("dracometeor")
    assert dm.self_boosts == {"spa": -2}

    # Rapid Spin (self +1 Spe)
    rs = dex.get_move("rapidspin")
    assert any(sec.get("self",{}).get("boosts")=={"spe":1} for sec in rs.secondaries)
    assert rs.is_contact is True

    # Chilling Water (target -1 Atk)
    cw = dex.get_move("chillingwater")
    assert any(sec.get("boosts")=={"atk":-1} for sec in cw.secondaries)
    assert cw.is_contact is False

    # Stealth Rock (reflectable by Magic Bounce)
    sr = dex.get_move("stealthrock")
    assert sr.is_reflectable is True

    # Toxic (reflectable by Magic Bounce)
    tox = dex.get_move("toxic")
    assert tox.is_reflectable is True

    # Earthquake (physical non-contact)
    eq = dex.get_move("earthquake")
    assert eq.is_contact is False
    assert eq.category == MoveCategory.PHYSICAL


def test_cartridge_stat_calculations():
    """Verify authentic cartridge stats computed for standard Gen 9 Pokemon."""
    dex = ShowdownDex.get_instance()

    # Ting-Lu level 100: Base HP 155 -> 451 HP with 0 EVs, 31 IVs
    tl = dex.calculate_pokemon_stats("Ting-Lu")
    assert tl["hp"] == 451

    # Great Tusk level 100: Base HP 115 -> 371 HP with 0 EVs, 31 IVs
    gt = dex.calculate_pokemon_stats("Great Tusk")
    assert gt["hp"] == 371

    # Gholdengo level 100: Base HP 87 -> 315 HP with 0 EVs, 31 IVs
    gh = dex.calculate_pokemon_stats("Gholdengo")
    assert gh["hp"] == 315

    # Kingambit level 100: Base HP 100 -> 341 HP with 0 EVs, 31 IVs
    kb = dex.calculate_pokemon_stats("Kingambit")
    assert kb["hp"] == 341

    # Shedinja level 100: Base HP 1 -> Always 1 HP
    shed = dex.calculate_pokemon_stats("Shedinja")
    assert shed["hp"] == 1

    # Competitive Great Tusk: Jolly nature (+spe, -spa), 252 Atk, 4 Def, 252 Spe
    comp_tusk = dex.calculate_pokemon_stats(
        "Great Tusk",
        evs={"hp": 0, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 252},
        nature="jolly"
    )
    assert comp_tusk["hp"] == 371
    assert comp_tusk["atk"] == 361
    assert comp_tusk["def"] == 299
    assert comp_tusk["spe"] == 300


def test_data_driven_drain_and_recoil_simulation():
    """Verify simulate_turn_transition handles drain and recoil generically via Move attributes."""
    dex = ShowdownDex.get_instance()

    bitter_blade = dex.get_move("bitterblade")
    brave_bird = dex.get_move("bravebird")
    splash = Move.create("Splash", PokemonType.NORMAL, MoveCategory.STATUS, 0)

    # 1. Drain Move: Bitter Blade restores 50% of damage dealt
    ceruledge = Pokemon(species="Ceruledge", current_hp=100, max_hp=300, moves=[bitter_blade])
    target = Pokemon(species="Target", current_hp=300, max_hp=300, moves=[splash])
    state = BattleState(
        p1=BattleSide(pokemon=[ceruledge], active_index=0),
        p2=BattleSide(pokemon=[target], active_index=0)
    )
    s_next = simulate_turn_transition(state, MoveAction("bitterblade", 1), MoveAction("splash", 1))
    assert s_next.p1.active_pokemon.current_hp > 100  # Healed from drain!

    # 2. Recoil Move: Brave Bird deals recoil to user
    corviknight = Pokemon(species="Corviknight", current_hp=300, max_hp=300, moves=[brave_bird])
    target2 = Pokemon(species="Target", current_hp=300, max_hp=300, moves=[splash])
    state2 = BattleState(
        p1=BattleSide(pokemon=[corviknight], active_index=0),
        p2=BattleSide(pokemon=[target2], active_index=0)
    )
    s2_next = simulate_turn_transition(state2, MoveAction("bravebird", 1), MoveAction("splash", 1))
    assert s2_next.p1.active_pokemon.current_hp < 300  # Took recoil damage!


def test_data_driven_self_and_target_boosts():
    """Verify self-boosts and stat drops apply dynamically based on move data."""
    dex = ShowdownDex.get_instance()

    close_combat = dex.get_move("closecombat")
    draco_meteor = dex.get_move("dracometeor")
    chilling_water = dex.get_move("chillingwater")
    splash = Move.create("Splash", PokemonType.NORMAL, MoveCategory.STATUS, 0)

    # Close Combat drops Def & SpD by 1
    tusk = Pokemon(species="Great Tusk", moves=[close_combat])
    opp = Pokemon(species="Blissey", moves=[splash])
    st1 = BattleState(p1=BattleSide(pokemon=[tusk], active_index=0), p2=BattleSide(pokemon=[opp], active_index=0))
    res1 = simulate_turn_transition(st1, MoveAction("closecombat", 1), MoveAction("splash", 1))
    assert res1.p1.active_pokemon.boosts["def"] == -1
    assert res1.p1.active_pokemon.boosts["spd"] == -1

    # Draco Meteor drops SpA by 2
    drag = Pokemon(species="Dragapult", moves=[draco_meteor])
    st2 = BattleState(p1=BattleSide(pokemon=[drag], active_index=0), p2=BattleSide(pokemon=[opp], active_index=0))
    res2 = simulate_turn_transition(st2, MoveAction("dracometeor", 1), MoveAction("splash", 1))
    assert res2.p1.active_pokemon.boosts["spa"] == -2

    # Chilling Water drops target Atk by 1
    drag_cw = Pokemon(species="Dragapult", moves=[chilling_water])
    st3 = BattleState(p1=BattleSide(pokemon=[drag_cw], active_index=0), p2=BattleSide(pokemon=[tusk], active_index=0))
    res3 = simulate_turn_transition(st3, MoveAction("chillingwater", 1), MoveAction("splash", 1))
    assert res3.p2.active_pokemon.boosts["atk"] == -1


def test_air_balloon_uniform_string_normalization():
    """Verify Air Balloon immune to spikes and pops on damage regardless of casing or spacing."""
    dex = ShowdownDex.get_instance()
    shadow_ball = dex.get_move("shadowball")
    splash = Move.create("Splash", PokemonType.NORMAL, MoveCategory.STATUS, 0)

    for item_str in ("Air Balloon", "airballoon", "air-balloon", "  Air Balloon  "):
        mon = Pokemon(species="Gholdengo", item=item_str, current_hp=300, max_hp=300)
        assert clean_key(mon.item) == "airballoon"

        # Check Spikes immunity
        side = BattleSide(pokemon=[mon], active_index=0, hazards={Hazard.SPIKES_1: 3})
        state = BattleState(p1=side, p2=BattleSide(pokemon=[Pokemon(species="Target")]))
        apply_entry_hazards(state, side_idx=1, mon_idx=0)
        assert mon.current_hp == 300  # Immune to Spikes!

        # Check popping on direct damage
        attacker = Pokemon(species="Dragapult", moves=[shadow_ball])
        battle = BattleState(p1=BattleSide(pokemon=[attacker], active_index=0), p2=BattleSide(pokemon=[mon], active_index=0))
        next_b = simulate_turn_transition(battle, MoveAction("shadowball", 1), MoveAction("splash", 1))
        assert next_b.p2.active_pokemon.item is None  # Balloon popped!


def test_spiky_shield_contact_distinction():
    """Verify Spiky Shield chips contact moves (Close Combat) but not non-contact moves (Earthquake)."""
    dex = ShowdownDex.get_instance()
    spiky_shield = dex.get_move("spikyshield")
    close_combat = dex.get_move("closecombat")
    earthquake = dex.get_move("earthquake")

    # 1. Close Combat (Contact: True) -> Attacker takes 1/8 chip
    ogerpon = Pokemon(species="Ogerpon-Wellspring", moves=[spiky_shield], protect_streak=0)
    tusk = Pokemon(species="Great Tusk", current_hp=371, max_hp=371, moves=[close_combat])
    st1 = BattleState(p1=BattleSide(pokemon=[ogerpon], active_index=0), p2=BattleSide(pokemon=[tusk], active_index=0))
    res1 = simulate_turn_transition(st1, MoveAction("spikyshield", 1), MoveAction("closecombat", 1))
    assert res1.p2.active_pokemon.current_hp < 371  # Took Spiky Shield contact penalty!

    # 2. Earthquake (Contact: False) -> Attacker takes NO chip
    ogerpon2 = Pokemon(species="Ogerpon-Wellspring", moves=[spiky_shield], protect_streak=0)
    tusk2 = Pokemon(species="Great Tusk", current_hp=371, max_hp=371, moves=[earthquake])
    st2 = BattleState(p1=BattleSide(pokemon=[ogerpon2], active_index=0), p2=BattleSide(pokemon=[tusk2], active_index=0))
    res2 = simulate_turn_transition(st2, MoveAction("spikyshield", 1), MoveAction("earthquake", 1))
    assert res2.p2.active_pokemon.current_hp == 371  # Did NOT take Spiky Shield penalty!