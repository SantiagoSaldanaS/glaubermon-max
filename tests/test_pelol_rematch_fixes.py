import pytest
from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, StatusCondition, ActionType, Hazard
from glaubermon.search.subgame_resolver import SubgameResolver
from glaubermon.search.evaluators import HeuristicEvaluator


def test_tinglu_rejects_redundant_stealth_rock():
    """Turn 17 Autopsy: Ting-Lu must NEVER click Stealth Rock when Stealth Rock is already active."""
    tinglu = Pokemon(
        species="Ting-Lu", types=(PokemonType.DARK, PokemonType.GROUND),
        raw_stats={"hp": 451, "atk": 256, "def": 286, "spa": 130, "spd": 287, "spe": 126},
        current_hp=400, max_hp=451, status=StatusCondition.NONE,
        moves=[
            Move(id="earthquake", name="Earthquake", move_type=PokemonType.GROUND, category=MoveCategory.PHYSICAL, base_power=100),
            Move(id="ruination", name="Ruination", move_type=PokemonType.DARK, category=MoveCategory.SPECIAL, base_power=1),
            Move(id="stealthrock", name="Stealth Rock", move_type=PokemonType.ROCK, category=MoveCategory.STATUS, base_power=0),
            Move(id="whirlwind", name="Whirlwind", move_type=PokemonType.NORMAL, category=MoveCategory.STATUS, base_power=0, priority=-6),
        ],
        item="leftovers", ability="Vessel of Ruin"
    )
    gliscor = Pokemon(
        species="Gliscor", types=(PokemonType.GROUND, PokemonType.FLYING),
        raw_stats={"hp": 354, "atk": 226, "def": 286, "spa": 113, "spd": 248, "spe": 226},
        current_hp=300, max_hp=354, status=StatusCondition.TOXIC,
        moves=[
            Move(id="earthquake", name="Earthquake", move_type=PokemonType.GROUND, category=MoveCategory.PHYSICAL, base_power=100),
            Move(id="toxic", name="Toxic", move_type=PokemonType.POISON, category=MoveCategory.STATUS, base_power=0),
            Move(id="protect", name="Protect", move_type=PokemonType.NORMAL, category=MoveCategory.STATUS, base_power=0, priority=4),
            Move(id="spikes", name="Spikes", move_type=PokemonType.GROUND, category=MoveCategory.STATUS, base_power=0),
        ],
        item="toxicorb", ability="Poison Heal"
    )
    # Opponent already has Stealth Rock active on their side!
    state = BattleState(
        p1=BattleSide(pokemon=[tinglu], active_index=0, hazards={}),
        p2=BattleSide(pokemon=[gliscor, gliscor.clone()], active_index=0, hazards={Hazard.STEALTH_ROCK: 1})
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    sr_strat = next((p for a, p in zip(actions, strat) if getattr(a, "move_id", "") == "stealthrock"), 0.0)
    assert sr_strat < 0.01, f"Stealth Rock must have ~0% probability when already active, got {sr_strat}"
    assert action.move_id != "stealthrock", f"Ting-Lu selected redundant Stealth Rock: {action.move_id}"


def test_kingambit_turn27_kowtow_cleave_over_nonlethal_sucker_punch():
    """Turn 27 Autopsy: Kingambit at +2 Atk vs 167 HP Kingambit must choose a lethal attack (Iron Head / Kowtow Cleave) over non-lethal Sucker Punch."""
    # Glaubermax Kingambit: +2 Atk, Tera Flying, 169 HP, Supreme Overlord (3 fallen)
    our_kingambit = Pokemon(
        species="Kingambit", types=(PokemonType.DARK, PokemonType.STEEL),
        raw_stats={"hp": 341, "atk": 405, "def": 276, "spa": 140, "spd": 206, "spe": 136},
        current_hp=169, max_hp=341, status=StatusCondition.NONE,
        moves=[
            Move(id="kowtowcleave", name="Kowtow Cleave", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=85),
            Move(id="suckerpunch", name="Sucker Punch", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=70, priority=1),
            Move(id="ironhead", name="Iron Head", move_type=PokemonType.STEEL, category=MoveCategory.PHYSICAL, base_power=80),
            Move(id="swordsdance", name="Swords Dance", move_type=PokemonType.NORMAL, category=MoveCategory.STATUS, base_power=0),
        ],
        boosts={"atk": 2}, item="leftovers", ability="Supreme Overlord",
        is_terastallized=True, tera_type=PokemonType.FLYING
    )
    # Pelol94 Kingambit: entered fresh on Turn 26, so +0 Atk, 167 HP, Supreme Overlord (3 fallen)
    opp_kingambit = Pokemon(
        species="Kingambit", types=(PokemonType.DARK, PokemonType.STEEL),
        raw_stats={"hp": 341, "atk": 405, "def": 276, "spa": 140, "spd": 206, "spe": 136},
        current_hp=167, max_hp=341, status=StatusCondition.NONE,
        moves=[
            Move(id="kowtowcleave", name="Kowtow Cleave", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=85),
            Move(id="suckerpunch", name="Sucker Punch", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=70, priority=1),
            Move(id="ironhead", name="Iron Head", move_type=PokemonType.STEEL, category=MoveCategory.PHYSICAL, base_power=80),
            Move(id="swordsdance", name="Swords Dance", move_type=PokemonType.NORMAL, category=MoveCategory.STATUS, base_power=0),
        ],
        boosts={"atk": 0}, item="leftovers", ability="Supreme Overlord"
    )

    def make_fainted():
        return Pokemon(species="Pikachu", current_hp=0, max_hp=100)

    p1_team = [our_kingambit, make_fainted(), make_fainted(), make_fainted()]
    p2_team = [opp_kingambit, make_fainted(), make_fainted(), make_fainted()]

    state = BattleState(
        p1=BattleSide(pokemon=p1_team, active_index=0),
        p2=BattleSide(pokemon=p2_team, active_index=0)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    sp_strat = next((p for a, p in zip(actions, strat) if getattr(a, "move_id", "") == "suckerpunch"), 0.0)

    assert action.move_id in ("ironhead", "kowtowcleave"), f"Must execute lethal attack, chose {action.move_id}"
    assert sp_strat < 0.05, f"Non-lethal Sucker Punch should have < 5% weight, got {sp_strat}"


def test_ogerpon_rejects_spiky_shield_into_setup():
    """Turn 29 Autopsy: Ogerpon must attack with Ivy Cudgel instead of clicking Spiky Shield against incoming Kingambit."""
    ogerpon = Pokemon(
        species="Ogerpon-Wellspring", types=(PokemonType.GRASS, PokemonType.WATER),
        raw_stats={"hp": 301, "atk": 279, "def": 204, "spa": 140, "spd": 228, "spe": 319},
        current_hp=301, max_hp=301, status=StatusCondition.NONE,
        moves=[
            Move(id="ivycudgel", name="Ivy Cudgel", move_type=PokemonType.WATER, category=MoveCategory.PHYSICAL, base_power=100),
            Move(id="hornleech", name="Horn Leech", move_type=PokemonType.GRASS, category=MoveCategory.PHYSICAL, base_power=75),
            Move(id="playrough", name="Play Rough", move_type=PokemonType.FAIRY, category=MoveCategory.PHYSICAL, base_power=90),
            Move(id="spikyshield", name="Spiky Shield", move_type=PokemonType.GRASS, category=MoveCategory.STATUS, base_power=0, priority=4),
        ],
        item="wellspringmask", ability="Water Absorb"
    )
    # Kingambit was sent out fresh on Turn 28
    kingambit = Pokemon(
        species="Kingambit", types=(PokemonType.DARK, PokemonType.STEEL),
        raw_stats={"hp": 341, "atk": 405, "def": 276, "spa": 140, "spd": 206, "spe": 136},
        current_hp=341, max_hp=341, status=StatusCondition.NONE,
        moves=[
            Move(id="swordsdance", name="Swords Dance", move_type=PokemonType.NORMAL, category=MoveCategory.STATUS, base_power=0),
            Move(id="kowtowcleave", name="Kowtow Cleave", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=85),
            Move(id="suckerpunch", name="Sucker Punch", move_type=PokemonType.DARK, category=MoveCategory.PHYSICAL, base_power=70, priority=1),
            Move(id="ironhead", name="Iron Head", move_type=PokemonType.STEEL, category=MoveCategory.PHYSICAL, base_power=80),
        ],
        boosts={"atk": 0}, item="leftovers", ability="Supreme Overlord"
    )
    state = BattleState(
        p1=BattleSide(pokemon=[ogerpon], active_index=0),
        p2=BattleSide(pokemon=[kingambit], active_index=0)
    )
    resolver = SubgameResolver(evaluator=HeuristicEvaluator())
    action, strat, actions, val = resolver.resolve_turn(state, depth=1, sample=False)

    ss_strat = next((p for a, p in zip(actions, strat) if getattr(a, "move_id", "") == "spikyshield"), 0.0)
    assert ss_strat < 0.10, f"Spiky Shield should have < 10% weight against setup/lethal threat, got {ss_strat}"
    assert action.move_id == "ivycudgel", f"Ogerpon should attack with Ivy Cudgel instead of giving free setup turn, chose: {action.move_id}"


def test_evaluator_spikes_grounded_scaling():
    """Verify that Spikes penalty scales with the count of grounded Pokémon on the team."""
    evaluator = HeuristicEvaluator()

    grounded_mons = [
        Pokemon(species="Ting-Lu", types=(PokemonType.DARK, PokemonType.GROUND), raw_stats={"hp": 451, "atk": 256, "def": 286, "spa": 130, "spd": 287, "spe": 126}, current_hp=451, max_hp=451),
        Pokemon(species="Great Tusk", types=(PokemonType.GROUND, PokemonType.FIGHTING), raw_stats={"hp": 371, "atk": 361, "def": 306, "spa": 127, "spd": 142, "spe": 273}, current_hp=371, max_hp=371),
        Pokemon(species="Kingambit", types=(PokemonType.DARK, PokemonType.STEEL), raw_stats={"hp": 341, "atk": 405, "def": 276, "spa": 140, "spd": 206, "spe": 136}, current_hp=341, max_hp=341),
        Pokemon(species="Gholdengo", types=(PokemonType.STEEL, PokemonType.GHOST), raw_stats={"hp": 315, "atk": 140, "def": 226, "spa": 365, "spd": 218, "spe": 204}, current_hp=315, max_hp=315),
    ]
    opp_mons = [
        Pokemon(species="Gliscor", types=(PokemonType.GROUND, PokemonType.FLYING), raw_stats={"hp": 354, "atk": 226, "def": 286, "spa": 113, "spd": 248, "spe": 226}, current_hp=354, max_hp=354),
        Pokemon(species="Dragapult", types=(PokemonType.DRAGON, PokemonType.GHOST), raw_stats={"hp": 317, "atk": 339, "def": 186, "spa": 236, "spd": 186, "spe": 421}, current_hp=317, max_hp=317),
    ]

    state_no_hazards = BattleState(
        p1=BattleSide(pokemon=grounded_mons, active_index=0, hazards={}),
        p2=BattleSide(pokemon=opp_mons, active_index=0, hazards={})
    )
    state_3_spikes = BattleState(
        p1=BattleSide(pokemon=grounded_mons, active_index=0, hazards={Hazard.SPIKES_1: 3}),
        p2=BattleSide(pokemon=opp_mons, active_index=0, hazards={})
    )

    score_clean = evaluator.evaluate(state_no_hazards)
    score_spikes = evaluator.evaluate(state_3_spikes)

    diff = score_clean - score_spikes
    # 3 layers of Spikes on a team of 4 grounded Pokémon must incur a significant penalty (> 0.10)
    assert diff >= 0.10, f"3 Spikes on 4 grounded Pokémon should heavily reduce score, diff={diff:.3f}"
