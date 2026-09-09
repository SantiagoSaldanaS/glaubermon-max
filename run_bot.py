import asyncio
import json
import logging
import websockets
import aiohttp
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveBot")

with open("showdown_config.json", "r") as f:
    cfg = json.load(f)

USERNAME = cfg["username"]
PASSWORD = cfg["password"]
ACTION_URL = "https://play.pokemonshowdown.com/action.php"
WS_URL = "wss://sim3.psim.us/showdown/websocket"

PACKED_META_TEAM = "]".join([
    "Great Tusk||boosterenergy|protosynthesis|rapidspin,closecombat,headlongrush,icespinner|Jolly|,,252,,4,252|||||Ice",
    "Gholdengo||airballoon|goodasgold|makeitrain,shadowball,nastyplot,recover|Timid|,,,252,4,252|||||Fighting",
    "Kingambit||blackglasses|supremeoverlord|kowtowcleave,suckerpunch,ironhead,swordsdance|Adamant|212,252,,,44,|||||Flying",
    "Dragapult||choicespecs|infiltrator|dracometeor,shadowball,flamethrower,uturn|Timid|,,,252,4,252|||||Ghost",
    "Ogerpon-Wellspring||wellspringmask|waterabsorb|ivycudgel,hornleech,playrough,spikyshield|Jolly|,,252,,4,252|||||Water",
    "Ting-Lu||leftovers|vesselofruin|stealthrock,earthquake,ruination,whirlwind|Impish|252,,4,,252,|||||Poison"
])

async def run():
    logger.info(f"Connecting to Showdown as {USERNAME}...")
    async with websockets.connect(WS_URL) as ws:
        async for msg in ws:
            lines = msg.split("\n")
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
                cmd = parts[1]

                if cmd == "challstr":
                    challstr = "|".join(parts[2:])
                    logger.info("Received challstr. Requesting login assertion...")
                    data = {"act": "login", "name": USERNAME, "pass": PASSWORD, "challstr": challstr}
                    async with aiohttp.ClientSession() as s:
                        async with s.post(ACTION_URL, data=data) as r:
                            text = await r.text()
                            if text.startswith("]"): text = text[1:]
                            auth = json.loads(text)
                            assertion = auth.get("assertion")
                            if assertion:
                                logger.info(f"Assertion obtained! Logging in as {USERNAME}...")
                                await ws.send(f"|/trn {USERNAME},0,{assertion}")
                            else:
                                logger.error(f"Login failed: {auth}")

                elif cmd == "updateuser":
                    user = parts[2].strip()
                    logger.info(f"Showdown User Confirmed: {user}")
                    if USERNAME.lower() in user.lower():
                        logger.info("Setting avatar and active tournament team...")
                        await ws.send("|/avatar red")
                        await ws.send(f"|/utm {PACKED_META_TEAM}")

                elif cmd == "pm":
                    sender = parts[2].strip()
                    text = "|".join(parts[4:])
                    logger.info(f"PM from {sender}: {text}")
                    if "/challenge" in text:
                        logger.info(f"Accepting challenge from {sender}...")
                        await ws.send(f"|/utm {PACKED_META_TEAM}")
                        await ws.send(f"|/accept {sender}")

                elif cmd in ("challenges", "updatechallenges"):
                    try:
                        c_data = json.loads("|".join(parts[2:]))
                        for challenger, fmt in c_data.get("challengesFrom", {}).items():
                            logger.info(f"Pending challenge from {challenger} in {fmt}. Accepting...")
                            await ws.send(f"|/utm {PACKED_META_TEAM}")
                            await ws.send(f"|/accept {challenger}")
                    except Exception as e:
                        logger.warning(f"Challenge parse error: {e}")

                elif cmd == "request" and room:
                    logger.info(f"Turn request in room {room}! Picking action...")
                    req = json.loads("|".join(parts[2:]))
                    if req.get("forceSwitch"):
                        await ws.send(f"{room}|/choose switch 1")
                    elif not req.get("wait"):
                        await ws.send(f"{room}|/choose move 1")

                elif cmd == "popup":
                    logger.warning(f"SHOWDOWN POPUP: {'|'.join(parts[2:])}")

asyncio.run(run())
