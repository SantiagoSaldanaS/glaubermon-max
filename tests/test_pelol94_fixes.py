import pytest
import numpy as np
from glaubermon.core.types import PokemonType, MoveCategory, ActionType, Hazard, StatusCondition
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.actions import MoveAction, SwitchAction
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.search.subgame_resolver import SubgameResolver, apply_entry_hazards
from glaubermon.inference.damage_calc import calculate_damage_rolls

@pytest.fixture
def resolver():
    return SubgameResolver()

def test_turn21_forced_switch_selects_kingambit(resolver):
    p2_mon = Pokemon(
        'dragapult',
        types=(PokemonType.DRAGON, PokemonType.GHOST),
        raw_stats={'hp': 317, 'atk': 279, 'def': 186, 'spa': 299, 'spd': 186, 'spe': 383},
        current_hp=165,
        max_hp=317,
        moves=[Move('shadowball', 'Shadow Ball', PokemonType.GHOST, MoveCategory.SPECIAL, 80, 1.0)],
        is_terastallized=True,
        tera_type=PokemonType.GHOST
    )
    p2 = BattleSide(pokemon=[p2_mon], active_index=0)

    fnt_mon = Pokemon('tinglu', types=(PokemonType.DARK, PokemonType.GROUND), raw_stats={'hp': 451, 'atk': 256, 'def': 286, 'spa': 131, 'spd': 196, 'spe': 126}, current_hp=0, max_hp=451, moves=[])
    kingambit = Pokemon(
        'kingambit',
        types=(PokemonType.DARK, PokemonType.STEEL),
        raw_stats={'hp': 341, 'atk': 379, 'def': 276, 'spa': 140, 'spd': 206, 'spe': 136},
        current_hp=341,
        max_hp=341,
        moves=[
            Move('suckerpunch', 'Sucker Punch', PokemonType.DARK, MoveCategory.PHYSICAL, 70, 1.0, priority=1),
            Move('kowtowcleave', 'Kowtow Cleave', PokemonType.DARK, MoveCategory.PHYSICAL, 85, 1.0)
        ],
        ability='supremeoverlord'
    )
    gholdengo = Pokemon(
        'gholdengo',
        types=(PokemonType.STEEL, PokemonType.GHOST),
        raw_stats={'hp': 315, 'atk': 140, 'def': 226, 'spa': 365, 'spd': 218, 'spe': 267},
        current_hp=315,
        max_hp=315,
        moves=[
            Move('shadowball', 'Shadow Ball', PokemonType.GHOST, MoveCategory.SPECIAL, 80, 1.0),
            Move('makeitrain', 'Make It Rain', PokemonType.STEEL, MoveCategory.SPECIAL, 120, 1.0)
        ]
    )

    state = BattleState(p1=BattleSide(pokemon=[fnt_mon, kingambit, gholdengo], active_index=0), p2=p2)

    s1 = state.clone()
    s1.p1.active_index = 1
    _, _, _, val_kingambit = resolver.resolve_turn(s1, depth=1, sample=False, last_action_was_switch=False)

    s2 = state.clone()
    s2.p1.active_index = 2
    _, _, _, val_gholdengo = resolver.resolve_turn(s2, depth=1, sample=False, last_action_was_switch=False)

    assert val_kingambit > val_gholdengo, f'Kingambit value ({val_kingambit}) must beat Gholdengo ({val_gholdengo})'
    assert val_kingambit > 15.0, f'Kingambit Sucker Punch should have massive advantage ({val_kingambit})'

def test_turn10_forced_switch_preserves_kingambit(resolver):
    tinglu_opp = Pokemon(
        'tinglu',
        types=(PokemonType.DARK, PokemonType.GROUND),
        raw_stats={'hp': 451, 'atk': 256, 'def': 286, 'spa': 131, 'spd': 196, 'spe': 126},
        current_hp=451,
        max_hp=451,
        moves=[
            Move('earthquake', 'Earthquake', PokemonType.GROUND, MoveCategory.PHYSICAL, 100, 1.0),
            Move('ruination', 'Ruination', PokemonType.DARK, MoveCategory.SPECIAL, 1, 0.9)
        ]
    )
    p2 = BattleSide(pokemon=[tinglu_opp], active_index=0)

    fnt = Pokemon('tinglu', types=(PokemonType.DARK, PokemonType.GROUND), raw_stats={'hp': 451, 'atk': 256, 'def': 286, 'spa': 131, 'spd': 196, 'spe': 126}, current_hp=0, max_hp=451, moves=[])
    kingambit = Pokemon('kingambit', types=(PokemonType.DARK, PokemonType.STEEL), raw_stats={'hp': 341, 'atk': 379, 'def': 276, 'spa': 140, 'spd': 206, 'spe': 136}, current_hp=341, max_hp=341, moves=[Move('kowtowcleave', 'Kowtow Cleave', PokemonType.DARK, MoveCategory.PHYSICAL, 85, 1.0)], ability='supremeoverlord')
    ogerpon = Pokemon('ogerponwellspring', types=(PokemonType.GRASS, PokemonType.WATER), raw_stats={'hp': 301, 'atk': 339, 'def': 204, 'spa': 140, 'spd': 228, 'spe': 350}, current_hp=301, max_hp=301, moves=[Move('ivycudgel', 'Ivy Cudgel', PokemonType.WATER, MoveCategory.PHYSICAL, 100, 1.0), Move('hornleech', 'Horn Leech', PokemonType.GRASS, MoveCategory.PHYSICAL, 75, 1.0)])
    tusk = Pokemon('greattusk', types=(PokemonType.GROUND, PokemonType.FIGHTING), raw_stats={'hp': 371, 'atk': 361, 'def': 301, 'spa': 127, 'spd': 142, 'spe': 273}, current_hp=371, max_hp=371, moves=[Move('closecombat', 'Close Combat', PokemonType.FIGHTING, MoveCategory.PHYSICAL, 120, 1.0)])

    state = BattleState(p1=BattleSide(pokemon=[fnt, kingambit, ogerpon, tusk], active_index=0), p2=p2)
    fallen = sum(1 for p in state.p1.pokemon if p.is_fainted)

    vals = {}
    for idx, mon in [(1, kingambit), (2, ogerpon), (3, tusk)]:
        s = state.clone()
        s.p1.active_index = idx
        _, _, _, val = resolver.resolve_turn(s, depth=1, sample=False, last_action_was_switch=False)
        if mon.species == 'kingambit' and fallen < 3:
            val -= 4.0 * (3 - fallen)
        vals[mon.species] = val

    assert vals['ogerponwellspring'] > vals['kingambit']
    assert vals['greattusk'] > vals['kingambit']

def test_candidate_subgame_prevents_unexpanded_rapid_spin(resolver):
    p2_mon = Pokemon(
        'dragapult',
        types=(PokemonType.DRAGON, PokemonType.GHOST),
        raw_stats={'hp': 317, 'atk': 279, 'def': 186, 'spa': 299, 'spd': 186, 'spe': 383},
        current_hp=165,
        max_hp=317,
        moves=[
            Move('dracometeor', 'Draco Meteor', PokemonType.DRAGON, MoveCategory.SPECIAL, 130, 0.9),
            Move('shadowball', 'Shadow Ball', PokemonType.GHOST, MoveCategory.SPECIAL, 80, 1.0)
        ]
    )
    p2 = BattleSide(pokemon=[p2_mon], active_index=0)

    tusk = Pokemon(
        'greattusk',
        types=(PokemonType.GROUND, PokemonType.FIGHTING),
        raw_stats={'hp': 371, 'atk': 361, 'def': 301, 'spa': 127, 'spd': 142, 'spe': 273},
        current_hp=200,
        max_hp=371,
        moves=[
            Move('headlongrush', 'Headlong Rush', PokemonType.GROUND, MoveCategory.PHYSICAL, 120, 1.0),
            Move('icespinner', 'Ice Spinner', PokemonType.ICE, MoveCategory.PHYSICAL, 80, 1.0),
            Move('rapidspin', 'Rapid Spin', PokemonType.NORMAL, MoveCategory.PHYSICAL, 50, 1.0),
            Move('stealthrock', 'Stealth Rock', PokemonType.ROCK, MoveCategory.STATUS, 0, 1.0)
        ]
    )
    p1 = BattleSide(pokemon=[tusk], active_index=0)
    state = BattleState(p1=p1, p2=p2)

    action, strat, actions, _ = resolver.resolve_turn(state, depth=2, sample=False)

    rapid_spin_p = 0.0
    for a, p in zip(actions, strat):
        if getattr(a, 'move_id', '') == 'rapidspin':
            rapid_spin_p = p

    assert rapid_spin_p == 0.0, f'Rapid Spin into Ghost-type must have 0% probability, got {rapid_spin_p}'
    assert getattr(action, 'move_id', '') != 'rapidspin', 'Must not choose Rapid Spin'

def test_tera_squander_penalty_on_debuffed_mon(resolver):
    p2_mon = Pokemon(
        'tinglu',
        types=(PokemonType.DARK, PokemonType.GROUND),
        raw_stats={'hp': 451, 'atk': 256, 'def': 286, 'spa': 131, 'spd': 196, 'spe': 126},
        current_hp=300,
        max_hp=451,
        moves=[Move('earthquake', 'Earthquake', PokemonType.GROUND, MoveCategory.PHYSICAL, 100, 1.0)]
    )
    p2 = BattleSide(pokemon=[p2_mon], active_index=0)

    dragapult = Pokemon(
        'dragapult',
        types=(PokemonType.DRAGON, PokemonType.GHOST),
        raw_stats={'hp': 317, 'atk': 279, 'def': 186, 'spa': 299, 'spd': 186, 'spe': 383},
        current_hp=120,
        max_hp=317,
        boosts={'spa': -2},
        moves=[Move('dracometeor', 'Draco Meteor', PokemonType.DRAGON, MoveCategory.SPECIAL, 130, 0.9)],
        tera_type=PokemonType.GHOST
    )
    p1 = BattleSide(pokemon=[dragapult], active_index=0, is_tera_used=False)
    state = BattleState(p1=p1, p2=p2)

    action, strat, actions, _ = resolver.resolve_turn(state, depth=1, sample=False)
    assert not getattr(action, 'is_tera', False), 'Debuffed Dragapult must not Terastallize'

def test_single_type_terastallized_damage_calc_index_safety():
    attacker = Pokemon(
        'dragapult',
        types=(PokemonType.GHOST,),
        raw_stats={'hp': 317, 'atk': 279, 'def': 186, 'spa': 299, 'spd': 186, 'spe': 383},
        current_hp=317,
        max_hp=317,
        is_terastallized=True
    )
    defender = Pokemon(
        'tinglu',
        types=(PokemonType.GROUND,),
        raw_stats={'hp': 451, 'atk': 256, 'def': 286, 'spa': 131, 'spd': 196, 'spe': 126},
        current_hp=451,
        max_hp=451,
        is_terastallized=True
    )
    move = Move('shadowball', 'Shadow Ball', PokemonType.GHOST, MoveCategory.SPECIAL, 80, 1.0)

    rolls = calculate_damage_rolls(attacker, defender, move)
    assert len(rolls) == 16
    assert all(r > 0 for r in rolls)


def test_forced_switch_combat_only_rejects_gholdengo_vs_ground(resolver):
    """When evaluated on combat moves only, Gholdengo without balloon gets heavily penalized vs Ground Ting-Lu,
    ensuring Ogerpon or Great Tusk is selected on forced switch."""
    tinglu_opp = Pokemon('tinglu', types=(PokemonType.DARK, PokemonType.GROUND), raw_stats={'hp': 514, 'atk': 256, 'def': 286, 'spa': 131, 'spd': 196, 'spe': 126}, current_hp=493, max_hp=514, moves=[
        Move('earthquake', 'Earthquake', PokemonType.GROUND, MoveCategory.PHYSICAL, 100, 1.0)
    ])
    p2 = BattleSide(pokemon=[tinglu_opp], active_index=0)

    gholdengo = Pokemon('gholdengo', types=(PokemonType.STEEL, PokemonType.GHOST), raw_stats={'hp': 315, 'atk': 140, 'def': 226, 'spa': 365, 'spd': 218, 'spe': 267}, current_hp=158, max_hp=315, moves=[
        Move('makeitrain', 'Make It Rain', PokemonType.STEEL, MoveCategory.SPECIAL, 120, 1.0)
    ], item=None)

    ogerpon = Pokemon('ogerponwellspring', types=(PokemonType.GRASS, PokemonType.WATER), raw_stats={'hp': 301, 'atk': 339, 'def': 204, 'spa': 140, 'spd': 228, 'spe': 350}, current_hp=301, max_hp=301, moves=[
        Move('ivycudgel', 'Ivy Cudgel', PokemonType.WATER, MoveCategory.PHYSICAL, 100, 1.0)
    ])

    dead_mon = Pokemon('tinglu', types=(PokemonType.DARK, PokemonType.GROUND), raw_stats={'hp': 514, 'atk': 256, 'def': 286, 'spa': 131, 'spd': 196, 'spe': 126}, current_hp=0, max_hp=514, moves=[])
    state = BattleState(p1=BattleSide(pokemon=[dead_mon, gholdengo, ogerpon], active_index=0), p2=p2)

    # Candidate 1: Gholdengo with combat moves only
    s1 = state.clone()
    s1.p1.active_index = 1
    m1 = [a for a in s1.get_valid_actions(1) if a.action_type == ActionType.MOVE]
    _, _, _, val1 = resolver.resolve_turn(s1, depth=1, sample=False, p1_actions_override=m1)

    # Candidate 2: Ogerpon with combat moves only
    s2 = state.clone()
    s2.p1.active_index = 2
    m2 = [a for a in s2.get_valid_actions(1) if a.action_type == ActionType.MOVE]
    _, _, _, val2 = resolver.resolve_turn(s2, depth=1, sample=False, p1_actions_override=m2)

    assert val2 > val1, f"Ogerpon combat value ({val2}) must beat Gholdengo ({val1})"
    assert val1 < 0, f"Gholdengo without balloon vs Ground Ting-Lu must have negative combat value ({val1})"


def test_turn22_voluntary_switch_rejects_early_kingambit(resolver):
    """Great Tusk facing Dragapult must not voluntarily switch to Kingambit when fallen < 3."""
    dead1 = Pokemon('tinglu', types=(PokemonType.DARK, PokemonType.GROUND), raw_stats={'hp': 514, 'atk': 256, 'def': 286, 'spa': 131, 'spd': 196, 'spe': 126}, current_hp=0, max_hp=514, moves=[])
    dead2 = Pokemon('dragapult', types=(PokemonType.DRAGON, PokemonType.GHOST), raw_stats={'hp': 317, 'atk': 279, 'def': 186, 'spa': 299, 'spd': 186, 'spe': 383}, current_hp=0, max_hp=317, moves=[])

    our_tusk = Pokemon('greattusk', types=(PokemonType.GROUND, PokemonType.FIGHTING), raw_stats={'hp': 371, 'atk': 361, 'def': 301, 'spa': 127, 'spd': 142, 'spe': 273}, current_hp=187, max_hp=371, status=StatusCondition.TOXIC, moves=[
        Move('headlongrush', 'Headlong Rush', PokemonType.GROUND, MoveCategory.PHYSICAL, 120, 1.0),
        Move('icespinner', 'Ice Spinner', PokemonType.ICE, MoveCategory.PHYSICAL, 80, 1.0)
    ])

    kingambit = Pokemon('kingambit', types=(PokemonType.DARK, PokemonType.STEEL), raw_stats={'hp': 341, 'atk': 379, 'def': 276, 'spa': 140, 'spd': 206, 'spe': 136}, current_hp=341, max_hp=341, moves=[
        Move('kowtowcleave', 'Kowtow Cleave', PokemonType.DARK, MoveCategory.PHYSICAL, 85, 1.0)
    ], ability='supremeoverlord')

    opp_pult = Pokemon('dragapult', types=(PokemonType.DRAGON, PokemonType.GHOST), raw_stats={'hp': 317, 'atk': 279, 'def': 186, 'spa': 299, 'spd': 186, 'spe': 383}, current_hp=278, max_hp=317, moves=[
        Move('dracometeor', 'Draco Meteor', PokemonType.DRAGON, MoveCategory.SPECIAL, 130, 0.9)
    ])

    p1 = BattleSide(pokemon=[our_tusk, dead1, dead2, kingambit], active_index=0)
    p2 = BattleSide(pokemon=[opp_pult], active_index=0)
    state = BattleState(p1=p1, p2=p2)

    act, strat, acts, _ = resolver.resolve_turn(state, depth=2, sample=False)
    for a, p in zip(acts, strat):
        if getattr(a, 'species', '') == 'kingambit':
            assert p == 0.0, f"Switch to Kingambit must have 0% probability, got {p}"
    assert getattr(act, 'species', '') != 'kingambit', "Must not switch to Kingambit"


def test_turn26_great_tusk_selects_close_combat_over_rapid_spin(resolver):
    """Against -2 Def opponent Great Tusk, Great Tusk must click Close Combat/Headlong Rush, not Rapid Spin."""
    opp_tusk = Pokemon('greattusk', types=(PokemonType.GROUND, PokemonType.FIGHTING), raw_stats={'hp': 371, 'atk': 361, 'def': 301, 'spa': 127, 'spd': 142, 'spe': 273}, current_hp=363, max_hp=371, moves=[
        Move('headlongrush', 'Headlong Rush', PokemonType.GROUND, MoveCategory.PHYSICAL, 120, 1.0),
        Move('closecombat', 'Close Combat', PokemonType.FIGHTING, MoveCategory.PHYSICAL, 120, 1.0)
    ], boosts={'def': -2, 'spd': -2})

    our_tusk = Pokemon('greattusk', types=(PokemonType.GROUND, PokemonType.FIGHTING), raw_stats={'hp': 371, 'atk': 361, 'def': 301, 'spa': 127, 'spd': 142, 'spe': 273}, current_hp=141, max_hp=371, moves=[
        Move('closecombat', 'Close Combat', PokemonType.FIGHTING, MoveCategory.PHYSICAL, 120, 1.0),
        Move('headlongrush', 'Headlong Rush', PokemonType.GROUND, MoveCategory.PHYSICAL, 120, 1.0),
        Move('rapidspin', 'Rapid Spin', PokemonType.NORMAL, MoveCategory.PHYSICAL, 50, 1.0)
    ])

    ogerpon = Pokemon('ogerponwellspring', types=(PokemonType.GRASS, PokemonType.WATER), raw_stats={'hp': 301, 'atk': 339, 'def': 204, 'spa': 140, 'spd': 228, 'spe': 350}, current_hp=16, max_hp=301, moves=[
        Move('hornleech', 'Horn Leech', PokemonType.GRASS, MoveCategory.PHYSICAL, 75, 1.0)
    ])

    p1 = BattleSide(pokemon=[our_tusk, ogerpon], active_index=0, hazards={Hazard.STEALTH_ROCK: 1, Hazard.SPIKES_1: 1})
    p2 = BattleSide(pokemon=[opp_tusk], active_index=0)
    state = BattleState(p1=p1, p2=p2)

    act, strat, acts, _ = resolver.resolve_turn(state, depth=2, sample=False)
    for a, p in zip(acts, strat):
        if getattr(a, 'move_id', '') == 'rapidspin':
            assert p == 0.0, f"Rapid Spin must have 0% probability, got {p}"
    assert getattr(act, 'move_id', '') in ('closecombat', 'headlongrush'), f"Must choose powerful STAB, got {act}"


def test_dragapult_choice_specs_pivots_at_minus_2_spa(resolver):
    """Dragapult locked into Draco Meteor at -2 SpA must pivot out to reset Choice Specs."""
    dragapult = Pokemon('dragapult', types=(PokemonType.DRAGON, PokemonType.GHOST), raw_stats={'hp': 317, 'atk': 279, 'def': 186, 'spa': 299, 'spd': 186, 'spe': 383}, current_hp=200, max_hp=317, boosts={'spa': -2}, moves=[
        Move('dracometeor', 'Draco Meteor', PokemonType.DRAGON, MoveCategory.SPECIAL, 130, 0.9)
    ], item='choicespecs', choice_locked_move='dracometeor')

    tinglu_opp = Pokemon('tinglu', types=(PokemonType.POISON,), raw_stats={'hp': 514, 'atk': 256, 'def': 286, 'spa': 131, 'spd': 196, 'spe': 126}, current_hp=300, max_hp=514, moves=[
        Move('earthquake', 'Earthquake', PokemonType.GROUND, MoveCategory.PHYSICAL, 100, 1.0)
    ])

    tusk = Pokemon('greattusk', types=(PokemonType.GROUND, PokemonType.FIGHTING), raw_stats={'hp': 371, 'atk': 361, 'def': 301, 'spa': 127, 'spd': 142, 'spe': 273}, current_hp=371, max_hp=371, moves=[
        Move('headlongrush', 'Headlong Rush', PokemonType.GROUND, MoveCategory.PHYSICAL, 120, 1.0)
    ])

    p1 = BattleSide(pokemon=[dragapult, tusk], active_index=0)
    p2 = BattleSide(pokemon=[tinglu_opp], active_index=0)
    state = BattleState(p1=p1, p2=p2)

    act, strat, acts, _ = resolver.resolve_turn(state, depth=1, sample=False)
    assert act.action_type == ActionType.SWITCH, f"Must switch out to reset -2 SpA choice lock, got {act}"