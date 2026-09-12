"""Official Pokémon Showdown WebSocket Bot Client for Glaubermon Max.

Connects directly to official Pokémon Showdown servers (play.pokemonshowdown.com)
via WebSockets, allowing you to challenge and play against Glaubermon Max
on the REAL Pokémon Showdown website.
"""

import os
import sys
import re
import json
import time
import random
import asyncio
import logging
from typing import Dict, Optional

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import aiohttp
import websockets

from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, ActionType, StatusCondition, Hazard, Weather, Terrain
from glaubermon.core.actions import Action, MoveAction, SwitchAction
from glaubermon.models.set_transformer import GlaubermonMaxNet
from glaubermon.search.evaluators import NeuralEvaluator, HeuristicEvaluator, HybridEvaluator
from glaubermon.search.subgame_resolver import SubgameResolver, apply_entry_hazards
from glaubermon.inference.damage_calc import calculate_damage_rolls
from glaubermon.inference.log_deducer import LogDeducer
from glaubermon.core.constants import get_type_effectiveness, compute_stat
from glaubermon.data.showdown_dex import ShowdownDex, STRING_TO_TYPE, STRING_TO_CATEGORY, clean_key
from glaubermon.data.meta_teams import (
    get_meta_team_balance,
    get_meta_team_hyper_offense,
    get_meta_team_stall,
    get_meta_team_pelol94,
    get_meta_pokemon_by_species,
    STANDARD_ITEMS
)

def clean_species_name(raw: str) -> str:
    """Clean Pokémon Showdown species names by stripping suffixes (-Tera, -Totem, etc.)."""
    if not raw:
        return ""
    name = raw.split(",")[0].strip()
    name = re.sub(r"-tera$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-totem$", "", name, flags=re.IGNORECASE)
    return name

def normalize_species_key(name: str) -> str:
    cleaned = clean_species_name(name)
    k = clean_key(cleaned)
    for base in ("ogerpon", "urshifu", "rotom", "landorus", "thundurus", "tornadus", "deoxys", "giratina", "zacian", "zamazenta"):
        if k.startswith(base):
            return base
    return k

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ShowdownBot")

SHOWDOWN_WS_URL = "wss://sim3.psim.us/showdown/websocket"
ACTION_URL = "https://play.pokemonshowdown.com/action.php"

PACKED_TEAMS: Dict[str, str] = {
    "balance": "]".join([
        "Great Tusk||boosterenergy|protosynthesis|closecombat,headlongrush,icespinner,rapidspin|Jolly|,252,,,4,252|||||,,,,,Ice",
        "Gholdengo||airballoon|goodasgold|makeitrain,shadowball,nastyplot,recover|Timid|,,,252,4,252|||||,,,,,Fighting",
        "Kingambit||blackglasses|supremeoverlord|kowtowcleave,suckerpunch,ironhead,swordsdance|Adamant|212,252,,,,44|||||,,,,,Flying",
        "Dragapult||choicespecs|infiltrator|dracometeor,shadowball,flamethrower,uturn|Timid|,,,252,4,252|||||,,,,,Ghost",
        "Ogerpon-Wellspring||wellspringmask|waterabsorb|ivycudgel,hornleech,playrough,spikyshield|Jolly|,252,,,4,252|||||,,,,,Water",
        "Ting-Lu||leftovers|vesselofruin|earthquake,ruination,stealthrock,whirlwind|Impish|252,,4,,252,|||||,,,,,Poison"
    ]),
    "hyper_offense": "]".join([
        "Iron Valiant||boosterenergy|quarkdrive|moonblast,closecombat,knockoff,thunderbolt|Timid|,,,252,4,252|||||,,,,,Fairy",
        "Roaring Moon||boosterenergy|protosynthesis|knockoff,dragondance,earthquake,acrobatics|Jolly|,252,,,4,252|||||,,,,,Flying",
        "Samurott-Hisui||focussash|sharpness|ceaselessedge,razorshell,knockoff,suckerpunch|Jolly|,252,,,4,252|||||,,,,,Ghost",
        "Great Tusk||heavydutyboots|protosynthesis|headlongrush,closecombat,icespinner,rapidspin|Jolly|,252,,,4,252|||||,,,,,Ice",
        "Kingambit||blackglasses|supremeoverlord|kowtowcleave,suckerpunch,ironhead,swordsdance|Adamant|212,252,,,,44|||||,,,,,Dark",
        "Gholdengo||choicescarf|goodasgold|makeitrain,shadowball,focusblast,trick|Timid|,,,252,4,252|||||,,,,,Steel"
    ]),
    "stall": "]".join([
        "Dondozo||leftovers|unaware|liquidation,bodypress,rest,sleeptalk|Impish|252,,252,,,4|||||,,,,,Grass",
        "Gliscor||toxicorb|poisonheal|earthquake,toxic,protect,spikes|Careful|244,,12,,252,|||||,,,,,Water",
        "Corviknight||leftovers|pressure|bravebird,bodypress,roost,defog|Impish|248,,252,,,8|||||,,,,,Dragon",
        "Heatran||leftovers|flashfire|magmastorm,earthpower,flashcannon,stealthrock|Calm|252,,,4,252,|||||,,,,,Grass",
        "Great Tusk||heavydutyboots|protosynthesis|rapidspin,knockoff,earthquake,icespinner|Impish|252,,252,,,4|||||,,,,,Water",
        "Kingambit||leftovers|supremeoverlord|kowtowcleave,suckerpunch,ironhead,swordsdance|Adamant|212,252,,,,44|||||,,,,,Flying"
    ]),
    "pelol": "]".join([
        "Iron Valiant||boosterenergy|quarkdrive|moonblast,closecombat,knockoff,thunderbolt|Timid|,,,252,4,252|||||,,,,,Fairy",
        "Gliscor||toxicorb|poisonheal|earthquake,toxic,protect,spikes|Careful|244,,12,,252,|||||,,,,,Water",
        "Great Tusk||boosterenergy|protosynthesis|closecombat,headlongrush,icespinner,rapidspin|Jolly|,252,,,4,252|||||,,,,,Ice",
        "Ting-Lu||leftovers|vesselofruin|earthquake,ruination,stealthrock,whirlwind|Impish|252,,4,,252,|||||,,,,,Poison",
        "Dragapult||choicespecs|infiltrator|dracometeor,shadowball,flamethrower,uturn|Timid|,,,252,4,252|||||,,,,,Ghost",
        "Kingambit||blackglasses|supremeoverlord|kowtowcleave,suckerpunch,ironhead,swordsdance|Adamant|212,252,,,,44|||||,,,,,Flying"
    ])
}
PACKED_TEAMS["ho"] = PACKED_TEAMS["hyper_offense"]
PACKED_TEAMS["pelol94"] = PACKED_TEAMS["pelol"]
PACKED_META_TEAM = PACKED_TEAMS["balance"]


def parse_hp_fraction(hp_str: str) -> float:
    """Parse Showdown HP string into a float fraction [0.0, 1.0]."""
    hp_str = hp_str.strip()
    if not hp_str or "fnt" in hp_str or hp_str == "0":
        return 0.0
    token = hp_str.split()[0]
    if "/" in token:
        try:
            cur, max_h = [float(x) for x in token.split("/")]
            return max(0.0, min(1.0, cur / max_h)) if max_h > 0 else 0.0
        except ValueError:
            return 1.0
    return 1.0


class ShowdownBot:
    """Live WebSocket Bot connecting to Pokémon Showdown."""

    def __init__(
        self,
        username: str = "GlaubermonAI",
        password: Optional[str] = None,
        depth: int = 2,
        team: str = "balance",
        target_challenge: Optional[str] = None,
        checkpoint: Optional[str] = None,
        ladder: bool = False,
        ladder_matches: int = 5,
        stealth: bool = True,
        evaluator: str = "hybrid",
        model: Optional[torch.nn.Module] = None,
        load_config: bool = True
    ):
        config_path = "showdown_config.json"
        if load_config and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if username == "GlaubermonAI" and "username" in cfg:
                        username = cfg["username"]
                    if password is None and "password" in cfg:
                        password = cfg["password"]
                    if "depth" in cfg:
                        depth = cfg["depth"]
                    if "team" in cfg and team == "balance":
                        team = cfg["team"]
                    if checkpoint is None and "checkpoint_path" in cfg:
                        checkpoint = cfg["checkpoint_path"]
                    if "ladder" in cfg and not ladder:
                        ladder = cfg["ladder"]
                    if "ladder_matches" in cfg and ladder_matches == 5:
                        ladder_matches = cfg["ladder_matches"]
                    if "stealth" in cfg:
                        stealth = cfg["stealth"]
                    if "evaluator" in cfg:
                        evaluator = cfg["evaluator"]
            except Exception:
                pass

        self.username = username
        self.password = password
        self.depth = depth
        self.team_choice = team.lower().replace("-", "_").strip()
        self.target_challenge = target_challenge
        self.checkpoint = checkpoint
        self.ladder_mode = ladder
        self.max_ladder_matches = ladder_matches
        self.stealth_mode = stealth
        self.evaluator_mode = (evaluator or "hybrid").lower().strip()
        self.ladder_games_played = 0
        self.ladder_wins = 0
        self.ladder_losses = 0
        self.searching_ladder = False
        self.announced_rooms: set = set()
        self.timer_started: set = set()
        self.dex = ShowdownDex.get_instance()

        if checkpoint is None:
            if os.path.exists("checkpoints/glaubermon_rebel_latest.pt"):
                checkpoint = "checkpoints/glaubermon_rebel_latest.pt"
            elif os.path.exists("checkpoints/glaubermon_alphazero_latest.pt"):
                checkpoint = "checkpoints/glaubermon_alphazero_latest.pt"
            else:
                checkpoint = "checkpoints/glaubermon_max_elite.pt"

        # Load Neural & Heuristic Evaluators into HybridEvaluator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model is not None else GlaubermonMaxNet(d_model=256, nhead=8, num_actions=14).to(self.device)
        if model is not None:
            self.device = next(model.parameters()).device
        ckpt_loaded = model is not None
        if model is not None:
            self.model.eval()
        elif checkpoint and os.path.exists(checkpoint):
            self.model.load_compatible_state_dict(torch.load(checkpoint, map_location=self.device))
            logger.info(f"Loaded AlphaZero/ReBeL weights from {checkpoint}")
            ckpt_loaded = True
        else:
            logger.warning(f"No checkpoint found at {checkpoint}; neural evaluation running without pre-trained weights")

        self.neural_eval = NeuralEvaluator(self.model, self.device)
        self.heuristic_eval = HeuristicEvaluator()

        if self.evaluator_mode == "neural" and ckpt_loaded:
            self.evaluator = self.neural_eval
            logger.info("Active Evaluator: NeuralEvaluator (100% AlphaZero/ReBeL neural value + policy prior)")
        elif self.evaluator_mode == "heuristic" or not ckpt_loaded:
            self.evaluator = self.heuristic_eval
            logger.info("Active Evaluator: HeuristicEvaluator (Domain-expert competitive heuristic)")
        else:
            # Default: hybrid (combines AlphaZero/ReBeL neural foresight + heuristic ground truth + policy priors)
            self.evaluator = HybridEvaluator(self.neural_eval, self.heuristic_eval, weight_neural=0.60)
            logger.info("Active Evaluator: HybridEvaluator (60% AlphaZero/ReBeL neural + 40% Heuristic + Policy Prior)")

        self.resolver = SubgameResolver(evaluator=self.evaluator)
        self.deducer = LogDeducer()

        self.ws = None
        self.active_battles: Dict[str, Dict] = {}
        self.opp_active: Dict[str, str] = {}
        self.opp_tera: Dict[str, str] = {}
        self.opp_tera_mon: Dict[str, str] = {}
        self.our_tera_used: Dict[str, bool] = {}
        self.our_side: Dict[str, str] = {}
        self.opp_team: Dict[str, list] = {}
        self.opp_moves: Dict[str, Dict[str, list]] = {}
        self.last_action_was_switch: Dict[str, bool] = {}
        self.sucker_punch_streak: Dict[str, int] = {}

        # Live Battle Tracking
        self.opp_hp: Dict[str, Dict[str, float]] = {}           # room -> species -> fraction (0.0 to 1.0)
        self.opp_fainted: Dict[str, set] = {}                  # room -> set of fainted species
        self.ident_to_species: Dict[str, Dict[str, str]] = {}  # room -> ident -> species
        self.p1_popped_items: Dict[str, set] = {}              # room -> set of species whose items popped
        self.opp_popped_items: Dict[str, set] = {}             # room -> set of opponent species whose items popped
        self.opp_boosts: Dict[str, Dict[str, Dict[str, int]]] = {}       # room -> species -> {stat: stage}
        self.our_boosts: Dict[str, Dict[str, Dict[str, int]]] = {}       # room -> species -> {stat: stage}
        self.side_hazards: Dict[str, Dict[str, Dict[Hazard, int]]] = {}   # room -> side_tag -> {hazard: count}
        self.mon_status: Dict[str, Dict[str, StatusCondition]] = {}       # room -> species -> status
        self.protect_streaks: Dict[str, Dict[str, int]] = {}              # room -> species -> streak count
        self.active_booster: Dict[str, Dict[tuple, str]] = {}  # room -> (side, species) -> stat
        self.room_weather: Dict[str, Weather] = {}                         # room -> Weather enum
        self.room_terrain: Dict[str, Terrain] = {}                         # room -> Terrain enum
        self.last_rqid: Dict[str, int] = {}
        self.room_turn: Dict[str, int] = {}
        self.public_fields = {}
        self.public_volatiles = {}
        self.active_species_by_side = {}
        self.revealed_items = {}
        self.revealed_abilities = {}
        self.active_rooms: set = set()

    def get_current_packed_team(self) -> str:
        """Return Showdown packed team string based on configured team archetype."""
        k = (self.team_choice or "balance").lower().replace("-", "_").strip()
        if k == "random":
            archetypes = ["balance", "hyper_offense", "stall", "pelol"]
            chosen = random.choice(archetypes)
            logger.info(f"Random team selected for battle: {chosen.upper()}")
            return PACKED_TEAMS[chosen]
        return PACKED_TEAMS.get(k, PACKED_TEAMS["balance"])

    async def _activate_timer(self, room: str):
        """Automatically activate Showdown battle timer with realistic human delay."""
        await asyncio.sleep(random.uniform(2.5, 4.5))
        if self.ws:
            logger.info(f"[{room}] Automatically activating battle timer (/timer on)")
            await self.ws.send(f"{room}|/timer on")

    async def _queue_ladder_match(self, initial_delay: float = 3.0):
        """Autonomous ladder matchmaking queue with anti-spam cooldowns and single-battle lock."""
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        # Ensure any active battle finishes before queueing for next ladder match
        while len(self.active_rooms) > 0:
            logger.info(f"[LADDER] Waiting for active battle(s) {self.active_rooms} to conclude before searching...")
            await asyncio.sleep(3.0)
        if self.ladder_games_played >= self.max_ladder_matches:
            logger.info(f"[LADDER] Session target matches reached ({self.max_ladder_matches}). Stopping matchmaking.")
            self.searching_ladder = False
            return
        logger.info(f"[LADDER] Queueing for Gen 9 OU ranked match ({self.ladder_games_played + 1}/{self.max_ladder_matches})...")
        await self.ws.send(f"|/utm {self.get_current_packed_team()}")
        await self.ws.send("|/search gen9ou")

    async def run(self):
        """Connect to Pokémon Showdown with automatic heartbeat keepalive and reconnection."""
        while True:
            try:
                logger.info(f"Connecting to Pokémon Showdown WebSocket: {SHOWDOWN_WS_URL} ...")
                async with websockets.connect(
                    SHOWDOWN_WS_URL,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10
                ) as websocket:
                    self.ws = websocket
                    logger.info("Connected to Pokémon Showdown!")
                    async for message in websocket:
                        await self.handle_message(message)
            except Exception as e:
                logger.warning(f"Showdown WebSocket disconnected ({e}). Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def handle_message(self, message: str):
        """Parse Showdown protocol messages."""
        lines = message.split("\n")
        room = ""
        if lines[0].startswith(">"):
            room = lines[0][1:]
            lines = lines[1:]

        for line in lines:
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue

            command = parts[1]
            if room:
                from glaubermon.client.field_tracker import PublicFieldTracker
                self.public_fields.setdefault(room, PublicFieldTracker()).ingest(parts)
                from glaubermon.client.volatile_tracker import PublicVolatileTracker
                self.public_volatiles.setdefault(room, PublicVolatileTracker()).ingest(parts)

            # 1. Handle Login Challenge String
            if command == "challstr":
                challstr = "|".join(parts[2:])
                await self.login(challstr)

            # 1b. Handle Updateuser confirmation
            elif command == "updateuser":
                user_logged = parts[2].strip()
                is_named = parts[3].strip() if len(parts) > 3 else "0"
                logger.info(f"Logged in user confirmation: {user_logged} (named={is_named})")

                clean_logged = re.sub(r"[^a-zA-Z0-9]", "", user_logged).lower()
                clean_target = re.sub(r"[^a-zA-Z0-9]", "", self.username).lower()

                # If a registered password was supplied, do NOT queue ladder or challenge as a guest
                if self.password and (is_named != "1" or clean_logged != clean_target):
                    logger.info(f"Waiting for full authentication as {self.username} before queueing matches (current: {user_logged})...")
                    continue

                if self.target_challenge:
                    logger.info(f"Sending challenge to {self.target_challenge} (gen9randombattle)...")
                    await self.ws.send(f"|/challenge {self.target_challenge}, gen9randombattle")
                elif self.ladder_mode and not self.searching_ladder:
                    self.searching_ladder = True
                    asyncio.create_task(self._queue_ladder_match(initial_delay=3.0))

            # 1c. Battle Room Initialization & Live URL Announcement
            elif command == "init" and room and room.startswith("battle-"):
                self.active_rooms.add(room)
                if room not in self.announced_rooms:
                    self.announced_rooms.add(room)
                    self.searching_ladder = False
                    watch_url = f"https://play.pokemonshowdown.com/{room}"
                    print("\n" + "=" * 70)
                    print("  [BATTLE] MATCH STARTED!")
                    print(f"  Room ID: {room}")
                    print(f"  WATCH LIVE URL: {watch_url}")
                    print(f"  Active Team: {self.team_choice.upper()}")
                    if self.ladder_mode:
                        print(f"  Ladder Match: {self.ladder_games_played + 1} of {self.max_ladder_matches}")
                    print("=" * 70 + "\n")
                    if room not in self.timer_started:
                        self.timer_started.add(room)
                        asyncio.create_task(self._activate_timer(room))

            # 2. Handle Private Message / Challenge
            elif command == "pm":
                sender = parts[2].strip()
                recipient = parts[3].strip()
                text = "|".join(parts[4:])

                clean_sender = re.sub(r"[^a-zA-Z0-9]", "", sender).lower()
                clean_user = re.sub(r"[^a-zA-Z0-9]", "", self.username).lower()
                if clean_sender == clean_user:
                    continue

                if "/challenge" in text:
                    if self.ladder_mode:
                        logger.info(f"Ignoring challenge from {sender} (Bot is currently in Ladder Mode).")
                        continue
                    if len(self.active_rooms) > 0:
                        logger.info(f"Declining challenge from {sender} (Already in active battle: {self.active_rooms}).")
                        continue
                    logger.info(f"Received battle challenge from: {sender}")
                    # Set active tournament team and auto-accept challenge
                    await self.ws.send(f"|/utm {self.get_current_packed_team()}")
                    await self.ws.send(f"|/accept {sender}")
                    logger.info(f"Accepted challenge from {sender}!")

            # 2b. Handle Updatechallenges (Pending challenges when logging in)
            elif command in ("updatechallenges", "challenges"):
                try:
                    c_data = json.loads("|".join(parts[2:]))
                    challenges_from = c_data.get("challengesFrom", {})
                    for challenger, format_id in challenges_from.items():
                        if self.ladder_mode:
                            logger.info(f"Ignoring pending challenge from {challenger} (Bot is in Ladder Mode).")
                            continue
                        if len(self.active_rooms) > 0:
                            logger.info(f"Ignoring pending challenge from {challenger} (Already in active battle).")
                            continue
                        logger.info(f"Received pending challenge from: {challenger} in {format_id}")
                        await self.ws.send(f"|/utm {self.get_current_packed_team()}")
                        await self.ws.send(f"|/accept {challenger}")
                        logger.info(f"Accepted pending challenge from {challenger}!")
                except Exception as e:
                    logger.warning(f"Error parsing challenges: {e}")

            # 3. Handle Battle Request (Turn Decisions) - Dispatched concurrently
            elif command == "request" and room:
                raw_json = "|".join(parts[2:])
                if raw_json.strip():
                    try:
                        req_data = json.loads(raw_json)
                        if req_data.get("wait"):
                            continue
                        rqid = req_data.get("rqid")
                        if rqid is not None and self.last_rqid.get(room) == rqid:
                            continue
                        if rqid is not None:
                            self.last_rqid[room] = rqid
                        asyncio.create_task(self.handle_battle_turn(room, req_data))
                    except json.JSONDecodeError:
                        pass

            if room and len(parts)>3 and command == "-ability":
                ident = parts[2]
                spec = self.ident_to_species.get(room,{}).get(ident,ident.split(":")[-1].strip())
                self.revealed_abilities.setdefault(room,{})[(ident[:2],spec)] = clean_key(parts[3])
            if room and any(p.startswith("[from] ability: ") for p in parts):
                ident = next((p.removeprefix("[of] ") for p in parts if p.startswith("[of] ")), parts[2])
                if ident.startswith(("p1", "p2")):
                    spec = self.ident_to_species.get(room,{}).get(ident,ident.split(":")[-1].strip())
                    ability = next(p.split("ability: ",1)[1] for p in parts if p.startswith("[from] ability: "))
                    self.revealed_abilities.setdefault(room,{})[(ident[:2],spec)] = clean_key(ability)

            # Forward to LogDeducer
            try:
                self.deducer.parse_line(line)
            except Exception:
                pass

            if room and command == "turn" and len(parts) > 2:
                self.room_turn[room] = int(parts[2])

            # 3a. Handle Player Identity
            if room and command == "player" and len(parts) > 3:
                side_tag = parts[2].strip()
                p_name = parts[3].strip()
                clean_user = re.sub(r"[^a-zA-Z0-9]", "", self.username).lower()
                clean_pname = re.sub(r"[^a-zA-Z0-9]", "", p_name).lower()
                if clean_pname == clean_user:
                    self.our_side[room] = side_tag
                    logger.info(f"[{room}] Identified our side as: {side_tag} ({p_name})")

            # 3b. Handle In-Battle Switch, Drag, and Ident Mapping
            elif room and command in ("switch", "drag") and len(parts) > 3:
                ident = parts[2].strip()  # e.g. "p1a: Great Tusk"
                details = parts[3].strip()  # e.g. "Great Tusk, L100"
                side_tag = ident[:2]
                mon_name = clean_species_name(details)

                if room not in self.ident_to_species:
                    self.ident_to_species[room] = {}
                previous = self.active_species_by_side.setdefault(room,{}).get(side_tag)
                self.active_species_by_side[room][side_tag] = mon_name
                boosts_store = self.our_boosts if side_tag==self.our_side.get(room,"p1") else self.opp_boosts
                for species in (previous,mon_name):
                    if species:
                        boosts_store.setdefault(room,{})[species] = {}
                if room in self.protect_streaks and side_tag in self.protect_streaks[room]:
                    for k in self.protect_streaks[room][side_tag]:
                        self.protect_streaks[room][side_tag][k] = 0
                    self.protect_streaks[room][side_tag][normalize_species_key(mon_name)] = 0

                self.ident_to_species[room][ident] = mon_name

                # Parse HP condition if present in parts[4]
                hp_fraction = 1.0
                if len(parts) > 4:
                    hp_fraction = parse_hp_fraction(parts[4])

                our_side = self.our_side.get(room)
                if not our_side or side_tag != our_side:
                    self.opp_active[room] = mon_name
                    if room not in self.opp_hp:
                        self.opp_hp[room] = {}
                    self.opp_hp[room][mon_name] = hp_fraction
                    if hp_fraction == 0.0:
                        if room not in self.opp_fainted:
                            self.opp_fainted[room] = set()
                        self.opp_fainted[room].add(mon_name)
                    logger.info(f"[{room}] Opponent Active Mon: {mon_name} (HP: {hp_fraction*100:.1f}%)")

            elif room and command == "-terastallize" and len(parts) > 3:
                ident = parts[2].strip()
                tera_type = parts[3].strip()
                side_tag = ident[:2]
                raw_spec = self.ident_to_species.get(room, {}).get(ident, ident.split(":")[-1].strip())
                spec = clean_species_name(raw_spec)
                our_side = self.our_side.get(room)
                if not our_side or side_tag != our_side:
                    self.opp_tera[room] = tera_type
                    self.opp_tera_mon[room] = spec
                    logger.info(f"[{room}] Opponent {spec} Terastallized to: {tera_type}")
                else:
                    self.our_tera_used[room] = True
                    logger.info(f"[{room}] Our {spec} Terastallized to: {tera_type}")

                # Embody Aspect immediately grants a stat boost upon Terastallization
                spec_clean = clean_key(spec)
                if "ogerpon" in spec_clean:
                    boost_dict = self.our_boosts if (self.our_side.get(room) == side_tag) else self.opp_boosts
                    if room not in boost_dict:
                        boost_dict[room] = {}
                    if spec not in boost_dict[room]:
                        boost_dict[room][spec] = {}
                    if "hearthflame" in spec_clean:
                        boost_dict[room][spec]["atk"] = min(6, boost_dict[room][spec].get("atk", 0) + 1)
                    elif "cornerstone" in spec_clean:
                        boost_dict[room][spec]["def"] = min(6, boost_dict[room][spec].get("def", 0) + 1)
                    elif "wellspring" in spec_clean:
                        boost_dict[room][spec]["spd"] = min(6, boost_dict[room][spec].get("spd", 0) + 1)
                    else:
                        boost_dict[room][spec]["spe"] = min(6, boost_dict[room][spec].get("spe", 0) + 1)

            elif room and command == "poke" and len(parts) > 3:
                side_tag = parts[2].strip()
                details = parts[3].strip()
                spec = clean_species_name(details)
                our_side = self.our_side.get(room)
                if not our_side or side_tag != our_side:
                    if room not in self.opp_team:
                        self.opp_team[room] = []
                    if spec not in self.opp_team[room]:
                        self.opp_team[room].append(spec)

            elif room and command == "move" and len(parts) > 3:
                ident = parts[2].strip()
                m_name = parts[3].strip()
                side_tag = ident[:2]
                raw_spec = self.ident_to_species.get(room, {}).get(ident, ident.split(":")[-1].strip())
                spec = clean_species_name(raw_spec)
                c_spec = normalize_species_key(spec)
                m_clean = clean_key(m_name)

                # Track consecutive protect streak accurately per side
                if room not in self.protect_streaks:
                    self.protect_streaks[room] = {}
                if side_tag not in self.protect_streaks[room]:
                    self.protect_streaks[room][side_tag] = {}
                if m_clean in ("protect", "spikyshield", "detect", "banefulbunker", "silktrap"):
                    self.protect_streaks[room][side_tag][c_spec] = self.protect_streaks[room][side_tag].get(c_spec, 0) + 1
                else:
                    self.protect_streaks[room][side_tag][c_spec] = 0

                our_side = self.our_side.get(room)
                if not our_side or side_tag != our_side:
                    active_mon = self.opp_active.get(room)
                    if active_mon:
                        if room not in self.opp_moves:
                            self.opp_moves[room] = {}
                        if active_mon not in self.opp_moves[room]:
                            self.opp_moves[room][active_mon] = []
                        if m_name not in self.opp_moves[room][active_mon]:
                            self.opp_moves[room][active_mon].append(m_name)

            elif room and command == "-fail" and len(parts) > 2:
                # Do NOT reset protect_streaks to 0 on -fail:
                # If Protect/Spiky Shield failed, the Pokémon still attempted Protect this turn.
                # Resetting streak to 0 here causes an infinite Protect failure spam oscillation.
                pass

            elif room and command in ("-damage", "-heal") and len(parts) > 3:
                ident = parts[2].strip()
                side_tag = ident[:2]
                hp_str = parts[3].strip()
                hp_fraction = parse_hp_fraction(hp_str)

                our_side = self.our_side.get(room)
                if not our_side or side_tag != our_side:
                    spec = self.ident_to_species.get(room, {}).get(ident)
                    if not spec:
                        spec = ident.split(":")[-1].strip()
                    spec = clean_species_name(spec)
                    if spec:
                        if room not in self.opp_hp:
                            self.opp_hp[room] = {}
                        self.opp_hp[room][spec] = hp_fraction
                        if hp_fraction <= 0.001 or "fnt" in hp_str:
                            if room not in self.opp_fainted:
                                self.opp_fainted[room] = set()
                            self.opp_fainted[room].add(spec)
                        logger.info(f"[{room}] Opponent {spec} HP: {hp_fraction*100:.1f}%")

            elif room and command == "faint" and len(parts) > 2:
                ident = parts[2].strip()
                side_tag = ident[:2]
                raw_spec = self.ident_to_species.get(room, {}).get(ident, ident.split(":")[-1].strip())
                spec = clean_species_name(raw_spec)
                c_fnt = normalize_species_key(spec)
                if room in self.protect_streaks and side_tag in self.protect_streaks[room]:
                    self.protect_streaks[room][side_tag][c_fnt] = 0

                our_side = self.our_side.get(room)
                if not our_side or side_tag != our_side:
                    if spec:
                        if room not in self.opp_fainted:
                            self.opp_fainted[room] = set()
                        self.opp_fainted[room].add(spec)
                        if room not in self.opp_hp:
                            self.opp_hp[room] = {}
                        self.opp_hp[room][spec] = 0.0
                        logger.info(f"[{room}] Opponent {spec} FAINTED!")

            elif room and command in ("-boost", "-unboost") and len(parts) > 4:
                ident = parts[2].strip()
                stat = parts[3].strip().lower()
                amount = int(parts[4].strip())
                side_tag = ident[:2]
                spec = self.ident_to_species.get(room, {}).get(ident, ident.split(":")[-1].strip())
                boost_dict = self.our_boosts if (self.our_side.get(room) == side_tag) else self.opp_boosts
                if room not in boost_dict:
                    boost_dict[room] = {}
                if spec not in boost_dict[room]:
                    boost_dict[room][spec] = {}
                delta = amount if command == "-boost" else -amount
                boost_dict[room][spec][stat] = max(-6, min(6, boost_dict[room][spec].get(stat, 0) + delta))
                logger.info(f"[{room}] Boost Update: {spec} {stat} is now {boost_dict[room][spec][stat]}")

            elif room and command in ("-clearallboost",):
                self.opp_boosts[room] = {}
                self.our_boosts[room] = {}
                logger.info(f"[{room}] All boosts cleared!")

            elif room and command in ("-clearboost",) and len(parts) > 2:
                ident = parts[2].strip()
                spec = self.ident_to_species.get(room, {}).get(ident, ident.split(":")[-1].strip())
                store = self.our_boosts if ident[:2]==self.our_side.get(room,"p1") else self.opp_boosts
                store.setdefault(room,{})[spec] = {}

            elif room and command == "-sidestart" and len(parts) > 3:
                side_ident = parts[2].strip()
                side_tag = side_ident[:2]
                effect = parts[3].strip().lower()
                if room not in self.side_hazards:
                    self.side_hazards[room] = {"p1": {}, "p2": {}}
                if side_tag not in self.side_hazards[room]:
                    self.side_hazards[room][side_tag] = {}
                if "stealth rock" in effect:
                    self.side_hazards[room][side_tag][Hazard.STEALTH_ROCK] = 1
                elif "toxic spikes" in effect:
                    self.side_hazards[room][side_tag][Hazard.TOXIC_SPIKES_1] = min(2, self.side_hazards[room][side_tag].get(Hazard.TOXIC_SPIKES_1, 0) + 1)
                elif "spikes" in effect:
                    self.side_hazards[room][side_tag][Hazard.SPIKES_1] = min(3, self.side_hazards[room][side_tag].get(Hazard.SPIKES_1, 0) + 1)
                elif "sticky web" in effect:
                    self.side_hazards[room][side_tag][Hazard.STICKY_WEB] = 1
                logger.info(f"[{room}] Hazards on {side_tag}: {self.side_hazards[room][side_tag]}")

            elif room and command == "-sideend" and len(parts) > 3:
                side_ident = parts[2].strip()
                side_tag = side_ident[:2]
                effect = parts[3].strip().lower()
                if room in self.side_hazards and side_tag in self.side_hazards[room]:
                    if "stealth rock" in effect:
                        self.side_hazards[room][side_tag].pop(Hazard.STEALTH_ROCK, None)
                    elif "toxic spikes" in effect:
                        self.side_hazards[room][side_tag].pop(Hazard.TOXIC_SPIKES_1, None)
                    elif "spikes" in effect:
                        self.side_hazards[room][side_tag].pop(Hazard.SPIKES_1, None)
                    elif "sticky web" in effect:
                        self.side_hazards[room][side_tag].pop(Hazard.STICKY_WEB, None)
                logger.info(f"[{room}] Hazards cleared on {side_tag}: {self.side_hazards.get(room, {}).get(side_tag, {})}")

            elif room and command == "-status" and len(parts) > 3:
                ident = parts[2].strip()
                st_str = parts[3].strip().lower()
                spec = self.ident_to_species.get(room, {}).get(ident, ident.split(":")[-1].strip())
                st_map = {
                    "brn": StatusCondition.BURN, "par": StatusCondition.PARALYSIS,
                    "psn": StatusCondition.POISON, "tox": StatusCondition.TOXIC,
                    "slp": StatusCondition.SLEEP, "frz": StatusCondition.FREEZE
                }
                cond = st_map.get(st_str, StatusCondition.NONE)
                if room not in self.mon_status:
                    self.mon_status[room] = {}
                self.mon_status[room][(ident[:2],spec)] = cond
                logger.info(f"[{room}] Status applied to {spec}: {cond.name}")

            elif room and command == "-curestatus" and len(parts) > 2:
                ident = parts[2].strip()
                spec = self.ident_to_species.get(room, {}).get(ident, ident.split(":")[-1].strip())
                if room in self.mon_status:
                    self.mon_status[room][(ident[:2],spec)] = StatusCondition.NONE

            elif room and command in ("-enditem", "-item") and len(parts) > 3:
                ident = parts[2].strip()
                item_name = parts[3].strip().lower()
                spec = self.ident_to_species.get(room,{}).get(ident,ident.split(":")[-1].strip())
                self.revealed_items.setdefault(room,{})[(ident[:2],spec)] = item_name if command == "-item" else None
                if command == "-item":
                    for store in (self.p1_popped_items,self.opp_popped_items):
                        for key in (spec,normalize_species_key(spec),clean_key(spec)):
                            store.get(room,set()).discard(key)
                side_tag = ident[:2]
                our_side = self.our_side.get(room)
                if our_side and side_tag == our_side:
                    if command == "-enditem":
                        spec = ident.split(":")[-1].strip()
                        if room not in self.p1_popped_items:
                            self.p1_popped_items[room] = set()
                        self.p1_popped_items[room].add(spec)
                        logger.info(f"[{room}] Our {spec}'s item {item_name} popped/consumed!")
                else:
                    if command == "-enditem":
                        raw_spec = self.ident_to_species.get(room, {}).get(ident, ident.split(":")[-1].strip())
                        spec = clean_species_name(raw_spec)
                        if room not in self.opp_popped_items:
                            self.opp_popped_items[room] = set()
                        self.opp_popped_items[room].add(spec)
                        self.opp_popped_items[room].add(normalize_species_key(spec))
                        self.opp_popped_items[room].add(clean_key(spec))
                        logger.info(f"[{room}] Opponent {spec}'s item {item_name} popped/consumed!")

            elif room and command == "-start" and len(parts) > 3:
                ident = parts[2].strip()
                effect = parts[3].strip().lower()
                spec = clean_species_name(self.ident_to_species.get(room, {}).get(ident, ident.split(":")[-1].strip()))
                if "protosynthesis" in effect or "quarkdrive" in effect:
                    for s_name in ("atk", "def", "spa", "spd", "spe"):
                        if s_name in effect:
                            if room not in self.active_booster:
                                self.active_booster[room] = {}
                            self.active_booster[room][(ident[:2], spec)] = s_name
                            logger.info(f"[{room}] Booster Volatile Active on {spec}: {s_name.upper()} boosted!")
                            break

            elif room and command == "-end" and len(parts) > 3:
                ident = parts[2].strip()
                effect = parts[3].strip().lower()
                spec = clean_species_name(self.ident_to_species.get(room, {}).get(ident, ident.split(":")[-1].strip()))
                if "protosynthesis" in effect or "quarkdrive" in effect:
                    if room in self.active_booster and (ident[:2], spec) in self.active_booster[room]:
                        del self.active_booster[room][(ident[:2], spec)]
                        logger.info(f"[{room}] Booster Volatile Expired on {spec}")

            elif room and command == "-weather" and len(parts) > 2:
                w_str = parts[2].strip().lower()
                w_enum = {"none":Weather.NONE,"sunnyday":Weather.SUN,"raindance":Weather.RAIN,
                          "sandstorm":Weather.SANDSTORM,"snow":Weather.SNOW,"hail":Weather.SNOW,
                          "desolateland":Weather.HARSH_SUN,"primordialsea":Weather.HEAVY_RAIN,
                          "deltastream":Weather.STRONG_WINDS}.get(w_str,Weather.NONE)
                self.room_weather[room] = w_enum
                logger.info(f"[{room}] Weather updated to: {w_enum.name}")

            elif room and command == "-fieldstart" and len(parts) > 2:
                f_str = parts[2].strip().lower()
                if "electric" in f_str:
                    t_enum = Terrain.ELECTRIC
                elif "grassy" in f_str:
                    t_enum = Terrain.GRASSY
                elif "misty" in f_str:
                    t_enum = Terrain.MISTY
                elif "psychic" in f_str:
                    t_enum = Terrain.PSYCHIC
                else:
                    continue  # Trick Room and other fields do not clear terrain.
                self.room_terrain[room] = t_enum
                logger.info(f"[{room}] Terrain active: {t_enum.name}")

            elif room and command == "-fieldend" and len(parts) > 2:
                f_str = parts[2].strip().lower()
                if any(k in f_str for k in ("terrain", "electric", "grassy", "misty", "psychic")):
                    self.room_terrain[room] = Terrain.NONE
                    logger.info(f"[{room}] Terrain ended (now NONE)")

            # 4. Error Handling / Fallback
            elif command == "error":
                err_msg = "|".join(parts[2:])
                logger.warning(f"[{room}] Showdown Error: {err_msg}")
                # Suppress infinite feedback loop on non-actionable or rate-limiting errors
                non_actionable = ("nothing to choose", "too late", "already started", "spam", "can't speak")
                if room and not any(k in err_msg.lower() for k in non_actionable):
                    if any(k in err_msg.lower() for k in ("can't move", "can't switch", "disabled", "trapped")):
                        logger.info(f"[{room}] Sending emergency fallback: /choose default")
                        await self.ws.send(f"{room}|/choose default")

            # 5. Battle End
            elif command in ("win", "tie") and room:
                is_win = (command == "win")
                winner = parts[2] if is_win and len(parts) > 2 else "Tie"
                clean_win = re.sub(r"[^a-zA-Z0-9]", "", winner).lower()
                clean_us = re.sub(r"[^a-zA-Z0-9]", "", self.username).lower()
                is_our_win = is_win and (clean_win == clean_us)
                logger.info(f"Battle {room} ended! Winner: {winner}")
                self.active_rooms.discard(room)

                # Automatically save replay to Showdown server so battle never expires
                clean_id = room.replace("battle-", "")
                replay_url = f"https://replay.pokemonshowdown.com/{clean_id}"
                try:
                    await self.ws.send(f"{room}|/savereplay")
                    logger.info(f"Requested replay save: {replay_url}")
                except Exception as e:
                    logger.warning(f"Failed to request replay save: {e}")

                if self.ladder_mode:
                    self.ladder_games_played += 1
                    if is_our_win:
                        self.ladder_wins += 1
                    else:
                        self.ladder_losses += 1

                    outcome_str = "[VICTORY]" if is_our_win else ("[TIE]" if not is_win else "[DEFEAT]")
                    print("\n" + "=" * 70)
                    print(f"  [LADDER RESULT] {outcome_str}")
                    print(f"  Match: {self.ladder_games_played}/{self.max_ladder_matches}")
                    win_pct = (self.ladder_wins / self.ladder_games_played) * 100
                    print(f"  Current Session Record: {self.ladder_wins}W - {self.ladder_losses}L ({win_pct:.1f}%)")
                    print(f"  Replay URL: {replay_url}")
                    print("=" * 70 + "\n")

                # Leave room with human delay (completely silent - no chat messages)
                if self.stealth_mode:
                    await asyncio.sleep(random.uniform(3.0, 5.5))
                await self.ws.send(f"{room}|/leave")

                # Queue next ladder match if session is still running
                if self.ladder_mode:
                    if self.ladder_games_played < self.max_ladder_matches:
                        cooldown = random.uniform(18.0, 30.0) if self.stealth_mode else 3.0
                        logger.info(f"[LADDER] Cooling down for {cooldown:.1f}s before next search (Stealth protection)...")
                        self.searching_ladder = True
                        asyncio.create_task(self._queue_ladder_match(initial_delay=cooldown))
                    else:
                        final_pct = (self.ladder_wins / self.max_ladder_matches) * 100
                        print("\n" + "=" * 70)
                        print(f"  [LADDER SESSION COMPLETED] ({self.max_ladder_matches} matches)")
                        print(f"  Final Record: {self.ladder_wins}W - {self.ladder_losses}L ({final_pct:.1f}%)")
                        print("=" * 70 + "\n")

    async def login(self, challstr: str):
        """Authenticate bot with Pokémon Showdown server."""
        logger.info(f"Authenticating as {self.username}...")
        clean_user = re.sub(r"[^a-zA-Z0-9]", "", self.username).lower()

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        assertion = None
        if self.password:
            data = {
                "act": "login",
                "name": self.username,
                "pass": self.password,
                "challstr": challstr,
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(ACTION_URL, data=data) as resp:
                    resp_text = await resp.text()
                    if resp_text.startswith("]"):
                        resp_text = resp_text[1:]
                    try:
                        auth_data = json.loads(resp_text)
                        assertion = auth_data.get("assertion")
                        if not assertion and "actionerror" in auth_data:
                            logger.error(f"Showdown Authentication Error: {auth_data['actionerror']}")
                    except Exception:
                        assertion = resp_text.strip()
        else:
            data = {
                "act": "getassertion",
                "userid": clean_user,
                "challstr": challstr,
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(ACTION_URL, data=data) as resp:
                    assertion = (await resp.text()).strip()

        if assertion and not assertion.startswith(";;"):
            await self.ws.send(f"|/trn {self.username},0,{assertion}")
            logger.info(f"Sent authentication for: {self.username}")
            await self.ws.send("|/avatar red")
            await self.ws.send(f"|/utm {self.get_current_packed_team()}")
            print("\n" + "=" * 70)
            print(f"  GLAUBERMON MAX BOT IS ONLINE ON POKÉMON SHOWDOWN!")
            print(f"  Target User: {self.username}")
            print(f"  Lookahead Depth: {self.depth} (Minimax Subgame Lookahead)")
            print(f"  Active Team Archetype: {self.team_choice.upper()}")
            print(f"  1. Go to https://play.pokemonshowdown.com in your browser")
            print(f"  2. Find or Challenge username: {self.username}")
            print(f"  3. Play directly on the REAL Pokémon Showdown client!")
            print("=" * 70 + "\n")
        else:
            err_msg = f"Failed to authenticate as {self.username}: {assertion}"
            logger.error(err_msg)
            if self.ladder_mode or self.password:
                await self.ws.close()
                raise RuntimeError(err_msg)

    def build_battle_state(self, room: str, req: Dict) -> BattleState:
        side_pokemon = req.get("side", {}).get("pokemon", [])
        active_moves = req.get("active", [{}])[0].get("moves", [])
        team_k = (self.team_choice or "balance").lower().replace("-", "_").strip()
        if team_k in ("hyper_offense", "ho"):
            meta_sample = get_meta_team_hyper_offense()
        elif team_k == "stall":
            meta_sample = get_meta_team_stall()
        elif team_k in ("pelol", "pelol94"):
            meta_sample = get_meta_team_pelol94()
        else:
            meta_sample = get_meta_team_balance()

        # Find which index in side_pokemon is active (Showdown singles: index with active=True or index 0)
        p1_active_idx = 0
        for i, p in enumerate(side_pokemon):
            if p.get("active", False):
                p1_active_idx = i
                break

        # 1. Build P1 (Our Team)
        p1_mons = []
        for i, p in enumerate(side_pokemon):
            spec = clean_species_name(p.get("details", ""))
            (t1, t2), base = self.dex.get_pokemon_info(spec)
            calc_stats = self.dex.calculate_pokemon_stats(spec)
            max_hp = calc_stats["hp"]
            cond = p.get("condition", "")
            is_fainted = "fnt" in cond
            curr_hp = 0 if is_fainted else max_hp
            if "/" in cond and not is_fainted:
                try:
                    curr_hp, max_hp = [int(x) for x in cond.split()[0].split("/")]
                except Exception:
                    pass

            is_active_mon = (i == p1_active_idx)
            m_list = active_moves if (is_active_mon and active_moves) else p.get("moves", [])
            moves = []
            for m in m_list:
                # Retain all move slots in exact cartridge order so move_slot = i + 1 aligns with Showdown slots (1-4)
                if isinstance(m, dict):
                    m_id = m.get("id", "")
                    move_obj = self.dex.get_move(m_id)
                    is_dis = bool(m.get("disabled", False))
                    pp_val = int(m.get("pp", 10))
                    # A temporary restriction does not consume the move's PP.
                    move_obj.pp = pp_val
                    move_obj.request_disabled = is_dis
                    if "maxpp" in m:
                        move_obj.max_pp = int(m["maxpp"])
                    moves.append(move_obj)
                else:
                    m_id = str(m)
                    moves.append(self.dex.get_move(m_id))

            matching = [m for m in meta_sample if clean_key(m.species) == clean_key(spec)]
            
            # Accurate Tera Type and Terastallized status extraction
            is_terastallized = bool(p.get("terastallized"))
            tera_type = None
            if is_terastallized and isinstance(p.get("terastallized"), str):
                tera_type = STRING_TO_TYPE.get(str(p["terastallized"]).lower())
            if not tera_type and p.get("teraType"):
                tera_type = STRING_TO_TYPE.get(str(p["teraType"]).lower())
            if not tera_type and matching:
                tera_type = matching[0].tera_type
            if not tera_type:
                tera_type = t1

            ability = p.get("ability") or p.get("baseAbility") or (matching[0].ability if matching else self.dex.get_pokemon_ability(spec))

            # Prevent Air Balloon zombie bug: if popped in protocol or empty in request, item is None
            p_item = p.get("item", None)
            if spec in self.p1_popped_items.get(room, set()) or p_item == "":
                item = None
            elif p_item:
                item = p_item
            elif matching:
                item = matching[0].item
            else:
                item = None

            if not moves and matching:
                moves = matching[0].moves

            # Boosts and Status
            boosts = dict(self.our_boosts.get(room, {}).get(spec, {}))
            status = StatusCondition.NONE
            if "brn" in cond:
                status = StatusCondition.BURN
            elif "par" in cond:
                status = StatusCondition.PARALYSIS
            elif "tox" in cond:
                status = StatusCondition.TOXIC
            elif "psn" in cond:
                status = StatusCondition.POISON
            elif "slp" in cond:
                status = StatusCondition.SLEEP
            elif "frz" in cond:
                status = StatusCondition.FREEZE

            c_spec = normalize_species_key(spec)
            our_tag = self.our_side.get(room, "p1")
            p1_streak = self.protect_streaks.get(room, {}).get(our_tag, {}).get(c_spec, 0)

            # Ingest True Level 100 Cartridge Stats from Showdown Request
            p_stats = p.get("stats", {})
            if p_stats and all(k in p_stats for k in ("atk", "def", "spa", "spd", "spe")):
                cartridge_stats = {
                    "hp": max_hp,
                    "atk": int(p_stats["atk"]),
                    "def": int(p_stats["def"]),
                    "spa": int(p_stats["spa"]),
                    "spd": int(p_stats["spd"]),
                    "spe": int(p_stats["spe"]),
                }
            elif matching and matching[0].raw_stats:
                cartridge_stats = dict(matching[0].raw_stats)
                cartridge_stats["hp"] = max_hp
            else:
                cartridge_stats = dict(calc_stats)
                cartridge_stats["hp"] = max_hp

            b_stat1 = self.active_booster.get(room, {}).get((self.our_side.get(room, "p1"), clean_species_name(spec)))
            mon = Pokemon(
                species=spec,
                types=(t1, t2),
                raw_stats=cartridge_stats,
                current_hp=curr_hp,
                max_hp=max_hp,
                status=status,
                moves=moves,
                boosts=boosts,
                item=item,
                ability=ability,
                tera_type=tera_type,
                is_terastallized=is_terastallized,
                protect_streak=p1_streak,
                booster_stat=b_stat1
            )
            p1_mons.append(mon)

        # 2. Build P2 (Opponent Team)
        opp_species_list = list(self.opp_team.get(room, []))
        opp_active_spec = self.opp_active.get(room, (opp_species_list[0] if opp_species_list else "Great Tusk"))
        if opp_active_spec not in opp_species_list:
            opp_species_list.append(opp_active_spec)

        p2_mons = []
        p2_active_idx = 0
        for i, raw_spec in enumerate(opp_species_list):
            spec = clean_species_name(raw_spec)
            (t1, t2), base = self.dex.get_pokemon_info(spec)
            known = self.opp_moves.get(room, {}).get(spec, [])
            moves = []
            if known:
                for m_name in known:
                    moves.append(self.dex.get_move(m_name))

            # Authoritative Meta Lookup across all OU species and archetypes
            meta_mon = get_meta_pokemon_by_species(spec)
            if not meta_mon:
                matching = [m for m in meta_sample if clean_key(m.species) == clean_key(spec)]
                if matching:
                    meta_mon = matching[0]

            if len(moves) < 4 and meta_mon:
                for mm in meta_mon.moves:
                    if not any(clean_key(m.id) == clean_key(mm.id) for m in moves):
                        moves.append(mm)
                        if len(moves) == 4:
                            break

            item = meta_mon.item if meta_mon else STANDARD_ITEMS.get(clean_key(spec), None)
            opp_popped = self.opp_popped_items.get(room, set())
            if spec in opp_popped or normalize_species_key(spec) in opp_popped or clean_key(spec) in opp_popped:
                item = None
            public_item_key = ("p2" if self.our_side.get(room,"p1")=="p1" else "p1",spec)
            if public_item_key in self.revealed_items.get(room,{}):
                item = self.revealed_items[room][public_item_key]
            tera_type = meta_mon.tera_type if meta_mon else t1

            # Opponent Tera Isolation: ONLY the specific Pokémon that executed Terastallization is Terastallized
            is_opp_mon_tera = False
            opp_tera_recorded = self.opp_tera_mon.get(room)
            if opp_tera_recorded and clean_key(opp_tera_recorded) == clean_key(spec):
                is_opp_mon_tera = True
                if room in self.opp_tera:
                    tera_type = STRING_TO_TYPE.get(self.opp_tera[room].lower(), tera_type)

            # Resolve Opponent Ability
            ded_ability = self.revealed_abilities.get(room,{}).get(("p2" if self.our_side.get(room,"p1")=="p1" else "p1",spec))
            if ded_ability:
                ability = ded_ability
            elif meta_mon and meta_mon.ability:
                ability = meta_mon.ability
            else:
                ability = self.dex.get_pokemon_ability(spec)

            # True Level 100 Cartridge Stats for Opponent
            if meta_mon and meta_mon.raw_stats:
                cartridge_stats = dict(meta_mon.raw_stats)
                max_hp = meta_mon.max_hp
            else:
                b_atk = base.get("atk", 80)
                b_spa = base.get("spa", 80)
                b_spe = base.get("spe", 80)
                is_phys = b_atk >= b_spa
                # Fast OU sweepers (spe >= 80) run +Spe nature (jolly/timid) on competitive ladder
                if b_spe >= 80:
                    nature = "jolly" if is_phys else "timid"
                else:
                    nature = "adamant" if is_phys else "modest"
                ev_spread = {"hp": 0, "atk": 252 if is_phys else 0, "def": 0, "spa": 0 if is_phys else 252, "spd": 0, "spe": 252}
                cartridge_stats = self.dex.calculate_pokemon_stats(spec, evs=ev_spread, nature=nature)
                max_hp = cartridge_stats["hp"]

            # Determine Opponent HP & Faint status accurately from protocol tracking
            hp_frac = self.opp_hp.get(room, {}).get(spec, 1.0)
            is_opp_fnt = (spec in self.opp_fainted.get(room, set())) or (hp_frac <= 0.001)
            curr_hp = 0 if is_opp_fnt else max(1, int(max_hp * hp_frac))

            # Robust boost lookup matching exact species or base form
            boosts = {}
            for b_k, b_v in self.opp_boosts.get(room, {}).items():
                if clean_key(b_k) == clean_key(spec) or clean_key(b_k).split("-")[0] == clean_key(spec).split("-")[0]:
                    boosts = dict(b_v)
                    break
            status = self.mon_status.get(room, {}).get(("p2" if self.our_side.get(room,"p1")=="p1" else "p1",spec), StatusCondition.NONE)
            c_spec2 = normalize_species_key(spec)
            opp_side_tag = "p2" if self.our_side.get(room, "p1") == "p1" else "p1"
            p2_streak = self.protect_streaks.get(room, {}).get(opp_side_tag, {}).get(c_spec2, 0)
            b_stat2 = self.active_booster.get(room, {}).get((opp_side_tag, clean_species_name(spec)))

            mon = Pokemon(
                species=spec,
                types=(t1, t2),
                raw_stats=cartridge_stats,
                current_hp=curr_hp,
                max_hp=max_hp,
                status=status,
                moves=moves,
                boosts=boosts,
                item=item,
                ability=ability,
                tera_type=tera_type,
                is_terastallized=is_opp_mon_tera,
                protect_streak=p2_streak,
                booster_stat=b_stat2
            )
            p2_mons.append(mon)
            if spec == opp_active_spec:
                p2_active_idx = i

        can_tera = bool(req.get("active", [{}])[0].get("canTerastallize", False)) if req.get("active") else False
        p1_tera_used = self.our_tera_used.get(room, False) or (not can_tera)
        for p in side_pokemon:
            if p.get("terastallized"):
                p1_tera_used = True
                break

        our_tag = self.our_side.get(room, "p1")
        opp_tag = "p2" if our_tag == "p1" else "p1"
        p1_hazards = dict(self.side_hazards.get(room, {}).get(our_tag, {}))
        p2_hazards = dict(self.side_hazards.get(room, {}).get(opp_tag, {}))

        from glaubermon.client.field_tracker import PublicFieldTracker
        fields = self.public_fields.get(room, PublicFieldTracker())
        p1_side = BattleSide(pokemon=p1_mons, active_index=p1_active_idx, hazards=p1_hazards, is_tera_used=p1_tera_used,
                             screens=dict(fields.screens[our_tag]), tailwind=fields.tailwind[our_tag])
        p2_tera_used = bool(self.opp_tera.get(room))
        p2_side = BattleSide(pokemon=p2_mons, active_index=p2_active_idx, hazards=p2_hazards, is_tera_used=p2_tera_used,
                             screens=dict(fields.screens[opp_tag]), tailwind=fields.tailwind[opp_tag])

        if p1_side.active_pokemon:
            act_k = normalize_species_key(p1_side.active_pokemon.species)
            p1_side.active_pokemon.protect_streak = self.protect_streaks.get(room, {}).get(our_tag, {}).get(act_k, 0)
        if p2_side.active_pokemon:
            opp_k = normalize_species_key(p2_side.active_pokemon.species)
            p2_side.active_pokemon.protect_streak = self.protect_streaks.get(room, {}).get(opp_tag, {}).get(opp_k, 0)

        from glaubermon.client.volatile_tracker import PublicVolatileTracker
        tracker = self.public_volatiles.get(room,PublicVolatileTracker())
        sides = {our_tag:(1,p1_side),opp_tag:(2,p2_side)}
        for tag,(_,side) in sides.items():
            for mon in side.pokemon:
                mon.volatiles = tracker.observations(tag,mon.species,sides)
                mon.last_move = tracker.last_moves.get((tag,clean_key(mon.species)))
                for flag in tracker.entry_once.get((tag,clean_key(mon.species)),set()):
                    setattr(mon,flag,True)
        if p1_side.active_pokemon and req.get("active",[{}])[0].get("trapped"):
            p1_side.active_pokemon.volatiles["request_trapped"] = {}

        weather = self.room_weather.get(room, Weather.NONE)
        terrain = self.room_terrain.get(room, Terrain.NONE)
        return BattleState(p1=p1_side, p2=p2_side, pending_switches=(1,) if any(req.get("forceSwitch",[])) else (),
                           weather=weather, terrain=terrain, turn=self.room_turn.get(room, 1), trick_room=fields.trick_room,
                           weather_turns=-1 if weather != Weather.NONE else 0, terrain_turns=-1 if terrain != Terrain.NONE else 0)

    def select_lead_order(self, room: str, req: Optional[Dict] = None) -> str:
        """Dynamically evaluate opponent team preview to select the optimal starting lead.

        Evaluates type matchups, hazard setups, and prevents early Kingambit waste.
        Returns a 6-digit permutation string (e.g. '512346' for Ogerpon-Wellspring lead).
        """
        opp_team = self.opp_team.get(room, [])
        our_pokemon = []
        if req and req.get("side", {}).get("pokemon"):
            for p in req["side"]["pokemon"]:
                details = p.get("details", "").split(",")[0].strip()
                our_pokemon.append(details)

        # Fallback to standard 6 slots if req did not include list
        if not our_pokemon:
            our_pokemon = ["Great Tusk", "Gholdengo", "Kingambit", "Dragapult", "Ogerpon-Wellspring", "Ting-Lu"]

        num_mons = len(our_pokemon)
        scores = {i + 1: 2.5 for i in range(num_mons)}

        opp_lower = [s.lower().replace("-", "").replace(" ", "") for s in opp_team] if opp_team else []

        for i, spec in enumerate(our_pokemon):
            k = clean_key(spec)
            slot = i + 1

            # 1. Kingambit rule: NEVER lead Kingambit before allies fall
            if "kingambit" in k:
                scores[slot] = -99.0
                continue

            # 2. Natural entry hazard leads get a bonus
            if k in ("samurotthisui", "glimmora", "tinglu"):
                scores[slot] += 2.0
            elif k in ("gliscor", "dondozo", "heatran"):
                scores[slot] += 1.0

            # 3. Dynamic matchup bonuses against opponent threats
            if opp_lower:
                if "greattusk" in opp_lower:
                    if "ogerpon" in k:
                        scores[slot] += 2.5
                    elif "dragapult" in k or "samurotthisui" in k:
                        scores[slot] += 1.5
                    elif "tinglu" in k:
                        scores[slot] -= 3.0

                if any(th in opp_lower for th in ("ironvaliant", "zamazenta", "urshifu", "keldeo")):
                    if "gholdengo" in k:
                        scores[slot] += 2.5
                    elif "tinglu" in k:
                        scores[slot] -= 3.5

                if any(th in opp_lower for th in ("gliscor", "tinglu", "landorustherian", "landorus", "garganacl", "hippowdon")):
                    if "ogerpon" in k:
                        scores[slot] += 2.0
                    elif "gholdengo" in k:
                        scores[slot] += 1.5

                if any(th in opp_lower for th in ("glimmora", "samurotthisui", "ribombee", "tinkaton", "deoxysspeed")):
                    if "gholdengo" in k:
                        scores[slot] += 3.0
                    elif "greattusk" in k:
                        scores[slot] += 2.0

                if any(th in opp_lower for th in ("roaringmoon", "cinderace", "meowscarada", "dragapult")):
                    if "dragapult" in k or "ironvaliant" in k:
                        scores[slot] += 2.0

                if any(th in opp_lower for th in ("pelipper", "barraskewda", "archaludon", "walkingwake")):
                    if "ogerpon" in k:
                        scores[slot] += 3.5
                    elif "dondozo" in k:
                        scores[slot] += 2.0

        best_lead = max(scores, key=scores.get)
        order = [best_lead] + [i for i in range(1, num_mons + 1) if i != best_lead]
        order_str = "".join(str(x) for x in order)
        logger.info(f"[{room}] Team Preview Lead Selection: Evaluated scores {scores} -> Selected Lead: Slot {best_lead} (Permutation: {order_str})")
        return order_str

    async def handle_battle_turn(self, room: str, req: Dict):
        """Pure Game Theory Subgame Resolution: Set-Transformer Neural Net + Nash Equilibrium."""
        if req.get("wait"):
            logger.info(f"[{room}] Waiting for opponent choice...")
            return

        our_side_id = req.get("side", {}).get("id", "p1")
        self.our_side[room] = our_side_id

        # Team Preview: Intelligent Lead Selection with canonical Showdown protocol
        if req.get("teamPreview") or req.get("requestType") == "team":
            if self.stealth_mode:
                preview_delay = random.uniform(4.5, 7.0)
                logger.info(f"[{room}] Stealth: Simulating human preview evaluation ({preview_delay:.1f}s)...")
                await asyncio.sleep(preview_delay)
            lead_order = self.select_lead_order(room, req)
            rqid = req.get("rqid")
            rqid_str = f"|{rqid}" if rqid else ""
            logger.info(f"[{room}] Team Preview: Sending lead order {lead_order}...")
            await self.ws.send(f"{room}|/team {lead_order}")
            return

        # Build full BattleState representation
        state = self.build_battle_state(room, req)
        p1_act = state.p1.active_pokemon
        p2_act = state.p2.active_pokemon
        valid_p1_acts = state.get_valid_actions(player=1)
        logger.info(f"[{room}] DEBUG P1 Active: {p1_act.species if p1_act else 'None'} (HP: {p1_act.current_hp if p1_act else 0}/{p1_act.max_hp if p1_act else 0}) [Tera Available: {not state.p1.is_tera_used}]")
        logger.info(f"[{room}] DEBUG P1 Moves ({len(p1_act.moves) if p1_act else 0}): {[m.id for m in p1_act.moves] if p1_act else []}")
        logger.info(f"[{room}] DEBUG P1 Valid Actions ({len(valid_p1_acts)}): {[str(a) for a in valid_p1_acts]}")
        logger.info(f"[{room}] DEBUG P2 Active: {p2_act.species if p2_act else 'None'} (HP: {p2_act.current_hp if p2_act else 0}/{p2_act.max_hp if p2_act else 0}) [Tera Available: {not state.p2.is_tera_used}]")
        logger.info(f"[{room}] DEBUG P2 Team: {[f'{p.species}({p.current_hp}/{p.max_hp})' for p in state.p2.pokemon]}")

        # Public requests identify who must replace, but do not reveal an enemy's
        # queued move. Evaluate the replacement from the public hypothesis only.
        if any(req.get("forceSwitch", [])):
            action, strategy, actions, value = await asyncio.to_thread(
                self.resolver.resolve_turn, state, 1, False)
            if action is None or action.action_type != ActionType.SWITCH:
                raise RuntimeError("Replacement phase produced no switch")
            self.last_action_was_switch[room] = False
            logger.info(f"[{room}] Replacement decision: {action}, value={value:.3f}")
            if self.stealth_mode:
                await asyncio.sleep(random.uniform(1.8,3.5))
            await self.ws.send(f"{room}|/choose switch {action.target_slot}")
            return

        p1_actions_override = None

        # Filter out disabled moves (e.g. Choice Specs/Band/Scarf lock, Taunt, Disable, Encore, 0 PP)
        active_req_moves = req.get("active", [{}])[0].get("moves", [])
        if active_req_moves:
            legal_move_slots = [
                idx + 1 for idx, m_info in enumerate(active_req_moves)
                if isinstance(m_info, dict) and not m_info.get("disabled", False) and m_info.get("pp", 10) > 0
            ]
            current_actions = p1_actions_override if p1_actions_override is not None else state.get_valid_actions(player=1)
            if legal_move_slots:
                filtered_actions = [
                    a for a in current_actions
                    if a.action_type != ActionType.MOVE or getattr(a, "move_slot", 0) in legal_move_slots
                ]
                if filtered_actions:
                    p1_actions_override = filtered_actions
                if len(legal_move_slots) == 1 and p1_act:
                    it = clean_key(p1_act.item)
                    if it in ("choicespecs", "choiceband", "choicescarf"):
                        p1_act.choice_locked_move = active_req_moves[legal_move_slots[0] - 1].get("id", "")
            else:
                switches_only = [a for a in current_actions if a.action_type == ActionType.SWITCH]
                p1_actions_override = [MoveAction("struggle",1)] + switches_only

        if req.get("active", [{}])[0].get("trapped"):
            p1_actions_override = [a for a in (p1_actions_override if p1_actions_override is not None else state.get_valid_actions(1))
                                   if a.action_type == ActionType.MOVE]

        # Solve simultaneous extensive-form turn using AlphaZero Neural Net & Nash Equilibrium (non-blocking thread)
        can_tera = req.get("active", [{}])[0].get("canTerastallize", False)
        sucker_streak = self.sucker_punch_streak.get(room, 0)
        last_switch = self.last_action_was_switch.get(room, False)
        t_turn_start = time.time()
        action, strategy, actions, val = await asyncio.to_thread(
            self.resolver.resolve_turn, state, self.depth, False, p1_actions_override, sucker_streak, last_switch
        )

        self.last_action_was_switch[room] = (action.action_type == ActionType.SWITCH)
        if getattr(action, "move_id", "") == "suckerpunch":
            self.sucker_punch_streak[room] = sucker_streak + 1
        else:
            self.sucker_punch_streak[room] = 0

        logger.info(f"[{room}] Nash Equilibrium Decision: {action} (Game Value: {val:.3f})")
        for a, p in zip(actions, strategy):
            if p > 0.05:
                logger.info(f"    - Strategy: {a} -> {p * 100:.1f}%")

        # Stealth humanized thinking time
        if self.stealth_mode:
            elapsed = time.time() - t_turn_start
            avail_actions = p1_actions_override if p1_actions_override is not None else valid_p1_acts
            if len(avail_actions) <= 1:
                target_delay = random.uniform(1.4, 2.2)
            else:
                target_delay = max(1.8, min(5.5, random.gauss(3.2, 0.7)))
            remaining = max(0.0, target_delay - elapsed)
            if remaining > 0:
                logger.info(f"[{room}] Stealth: Human thinking delay ({remaining:.2f}s remaining of {target_delay:.2f}s)...")
                await asyncio.sleep(remaining)

        if action.action_type == ActionType.MOVE:
            tera_flag = " terastallize" if getattr(action, "is_tera", False) and can_tera else ""
            await self.ws.send(f"{room}|/choose move {action.move_slot}{tera_flag}")
        elif action.action_type == ActionType.SWITCH:
            chosen_spec = getattr(action, "species", None)
            target = getattr(action, "target_slot", 2)
            for slot_idx, p in enumerate(req.get("side", {}).get("pokemon", [])):
                cond = p.get("condition", "")
                if not p.get("active", False) and "fnt" not in cond and not p.get("fainted", False):
                    spec = p.get("details", "").split(",")[0].strip()
                    if chosen_spec and clean_key(spec) == clean_key(chosen_spec):
                        target = slot_idx + 1
                        break
            await self.ws.send(f"{room}|/choose switch {target}")
        else:
            await self.ws.send(f"{room}|/choose default")


import argparse

if __name__ == "__main__":
    default_user = "Glaubermax"
    default_pass = None
    default_depth = 2
    default_team = "balance"
    default_ckpt = None
    default_evaluator = "hybrid"

    config_path = "showdown_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                default_user = cfg.get("username", default_user)
                default_pass = cfg.get("password", default_pass)
                default_depth = cfg.get("depth", default_depth)
                default_team = cfg.get("team", default_team)
                default_ckpt = cfg.get("checkpoint_path", default_ckpt)
                default_evaluator = cfg.get("evaluator", default_evaluator)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Glaubermon Max Pokémon Showdown Bot")
    parser.add_argument("username", nargs="?", default=default_user, help="Bot username on Pokémon Showdown (default: Glaubermax)")
    parser.add_argument("--password", type=str, default=default_pass, help="Password if username is registered")
    parser.add_argument("--depth", type=int, default=default_depth, choices=[1, 2, 3, 4], help="Lookahead depth: 1 (1-ply), 2 (2-ply Nash minimax), 3 (3-ply beam lookahead), or 4 (default: 2)")
    parser.add_argument("--team", type=str, default=default_team, choices=["balance", "hyper_offense", "ho", "stall", "pelol", "pelol94", "random"], help="Team archetype: 'balance', 'hyper_offense'/'ho', 'stall', 'pelol', or 'random' (default: balance)")
    parser.add_argument("--challenge", type=str, default=None, help="Automatically send battle challenge to user")
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to checkpoint file")
    parser.add_argument("--evaluator", type=str, default=default_evaluator, choices=["hybrid", "neural", "heuristic"], help="Evaluator mode: 'hybrid' (default: 60%% AlphaZero neural + 40%% heuristic + policy prior), 'neural' (pure neural), or 'heuristic' (rule-based baseline)")
    parser.add_argument("--ladder", action="store_true", help="Enable autonomous ranked ladder matchmaking (/search gen9ou)")
    parser.add_argument("--ladder-matches", type=int, default=5, help="Number of ladder matches to play in this session (default: 5)")
    parser.add_argument("--no-stealth", action="store_true", help="Disable anti-detection humanized delays")
    args = parser.parse_args()

    bot = ShowdownBot(
        username=args.username,
        password=args.password,
        depth=args.depth,
        team=args.team,
        target_challenge=args.challenge,
        checkpoint=args.checkpoint,
        ladder=args.ladder,
        ladder_matches=args.ladder_matches,
        stealth=not args.no_stealth,
        evaluator=args.evaluator
    )
    asyncio.run(bot.run())
