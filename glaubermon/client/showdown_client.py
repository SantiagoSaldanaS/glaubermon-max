"""Asynchronous WebSocket Client for Pokémon Showdown integration."""

import asyncio
import json
import logging
from typing import Optional
import websockets
from glaubermon.inference.log_deducer import LogDeducer
from glaubermon.search.subgame_resolver import SubgameResolver

logger = logging.getLogger("Glaubermon MaxClient")


class ShowdownClient:
    """Connects to Pokémon Showdown servers to play automated competitive matches."""

    def __init__(
        self,
        username: str = "Glaubermon MaxAI",
        password: Optional[str] = None,
        server_uri: str = "ws://localhost:8000/showdown/websocket"
    ):
        self.username = username
        self.password = password
        self.server_uri = server_uri
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.deducer = LogDeducer()
        self.resolver = SubgameResolver()
        self.active_battles = {}

    async def connect(self):
        """Establish WebSocket connection with the Pokémon Showdown server."""
        logger.info(f"Connecting to Showdown server at {self.server_uri}...")
        try:
            async with websockets.connect(self.server_uri) as websocket:
                self.ws = websocket
                logger.info("Connected successfully.")
                async for message in websocket:
                    await self._handle_message(message)
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")

    async def _handle_message(self, raw_message: str):
        """Parse incoming multiline Showdown protocol messages."""
        lines = raw_message.split("\n")
        room = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                room = line[1:]
                continue

            parts = line.split("|")
            if len(parts) < 2:
                continue
            msg_type = parts[1]

            if msg_type == "challstr":
                # Challenge string received; handle login
                challstr = "|".join(parts[2:])
                await self._login(challstr)
            elif msg_type == "request":
                # Decision prompt for the bot
                if len(parts) >= 3 and parts[2].strip():
                    try:
                        req_data = json.loads(parts[2])
                        await self._handle_request(room, req_data)
                    except json.JSONDecodeError:
                        pass
            elif msg_type in ("-damage", "move", "switch", "-item", "-ability"):
                # Forward to real-time log deducer
                self.deducer.parse_line(line)

    async def _login(self, challstr: str):
        """Perform guest or authenticated login."""
        logger.info(f"Logging in as {self.username}...")
        if not self.password:
            # Guest login
            await self.send_message(f"|/trn {self.username},0,1")

    async def _handle_request(self, room: str, req_data: dict):
        """Evaluate request and send chosen action."""
        if req_data.get("wait"):
            return

        # Basic fallback move command (e.g. choose move 1)
        if "active" in req_data and req_data["active"]:
            cmd = "/choose move 1"
        elif "forceSwitch" in req_data:
            cmd = "/choose switch 2"
        else:
            cmd = "/choose default"

        await self.send_message(f"{room}|{cmd}")

    async def send_message(self, message: str):
        """Send command over the WebSocket connection."""
        if self.ws:
            await self.ws.send(message)
