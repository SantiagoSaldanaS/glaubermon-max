"""Official-Style Pokémon Showdown Battle Stadium for Glaubermon Max.

Replicates the 1:1 authentic Pokémon Showdown web client interface:
- Official Showdown field textures & battle platforms.
- 3D animated front & back sprites with grounded battle shadows.
- Authentic Showdown HP bars, level badges, and Pokéball status icons.
- Official 18-type color palettes for move buttons.
- Selectable real 6v6 Smogon Gen 9 OU Tournament Meta Teams (Balance, Hyper Offense, Stall).
- Live AI Nash Equilibrium Strategy Inspector.
"""

import os
import re
import json
import webbrowser
from typing import Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from glaubermon.core.battle_state import BattleState, BattleSide
from glaubermon.core.pokemon import Pokemon, Move
from glaubermon.core.types import PokemonType, MoveCategory, ActionType, Hazard
from glaubermon.core.actions import Action, MoveAction, SwitchAction
from glaubermon.models.set_transformer import GlaubermonMaxNet
from glaubermon.search.evaluators import NeuralEvaluator
from glaubermon.search.subgame_resolver import SubgameResolver, simulate_turn_transition
from glaubermon.data.meta_teams import create_meta_battle, META_TEAMS


app = FastAPI(title="Glaubermon Max: Official Showdown Stadium")

SESSION = {
    "state": None,
    "resolver": None,
    "logs": [],
    "last_ai_strat": [],
    "last_eval": 0.0,
    "p1_team_name": "balance",
    "p2_team_name": "balance"
}


def clean_mon_id(name: str) -> str:
    """Format species for Showdown animated sprite URL."""
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()


def serialize_battle_state(state: BattleState) -> Dict:
    """Format entire battle state for frontend rendering."""
    p1 = state.p1
    p2 = state.p2

    def format_side(side: BattleSide, is_human: bool):
        active = side.active_pokemon
        roster = []
        for i, mon in enumerate(side.pokemon):
            clean_id = clean_mon_id(mon.species)
            roster.append({
                "slot": i + 1,
                "species": mon.species,
                "sprite": f"https://play.pokemonshowdown.com/sprites/ani/{clean_id}.gif",
                "back_sprite": f"https://play.pokemonshowdown.com/sprites/ani-back/{clean_id}.gif",
                "icon": f"https://play.pokemonshowdown.com/sprites/gen5/{clean_id}.png",
                "hp": mon.current_hp,
                "max_hp": mon.max_hp,
                "hp_percent": int(mon.hp_percent * 100),
                "types": [mon.types[0].value] + ([mon.types[1].value] if mon.types[1] else []),
                "tera_type": mon.tera_type.value if mon.tera_type else None,
                "is_tera": mon.is_terastallized,
                "is_fainted": mon.is_fainted,
                "is_active": (i == side.active_index),
                "item": mon.item or "None",
                "boosts": {k.upper(): v for k, v in mon.boosts.items() if v != 0}
            })

        moves = []
        if active and not active.is_fainted:
            for i, m in enumerate(active.moves):
                moves.append({
                    "slot": i + 1,
                    "name": m.name,
                    "type": m.move_type.value,
                    "category": m.category.value,
                    "bp": m.base_power,
                    "accuracy": int(m.accuracy * 100),
                    "pp": m.pp
                })

        return {
            "active": roster[side.active_index] if active else None,
            "pokemon": roster,
            "moves": moves,
            "is_tera_used": side.is_tera_used,
            "switches": [idx + 1 for idx in side.available_switches()]
        }

    return {
        "turn": state.turn,
        "is_game_over": state.is_game_over,
        "winner": state.winner,
        "player": format_side(p1, is_human=True),
        "ai": format_side(p2, is_human=False),
        "logs": SESSION["logs"][-20:],
        "ai_analysis": SESSION["last_ai_strat"],
        "eval_value": round(SESSION["last_eval"], 3),
        "p1_team": SESSION["p1_team_name"].replace("_", " ").title(),
        "p2_team": SESSION["p2_team_name"].replace("_", " ").title(),
    }


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Pokémon Showdown - Glaubermon Max SOTA Arena</title>
    <style>
        :root {
            --ps-bg: #2b3e5a;
            --ps-border: #4a6382;
            --ps-card: rgba(22, 34, 52, 0.92);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: Verdana, 'Segoe UI', Tahoma, sans-serif; }
        body { background: #1a2536; color: #ffffff; display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 15px; }

        /* Main Showdown Window Wrapper */
        .showdown-client {
            width: 1040px;
            background: #24334a;
            border: 3px solid var(--ps-border);
            border-radius: 8px;
            box-shadow: 0 25px 60px rgba(0,0,0,0.8);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* Top Header Navigation */
        .topbar {
            background: #182333;
            padding: 8px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--ps-border);
            font-size: 13px;
        }
        .topbar-title { font-weight: bold; color: #73b9ff; font-size: 15px; }
        .team-selector { display: flex; gap: 10px; align-items: center; }
        .select-input { background: #2d3f59; border: 1px solid #4a6382; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }

        /* Battle Stage & Chat Container */
        .main-stage { display: flex; width: 100%; height: 460px; }

        /* 1:1 Showdown Battle Canvas */
        .battle-canvas {
            width: 660px;
            height: 100%;
            position: relative;
            background: radial-gradient(ellipse at 50% 60%, #4a8505 0%, #2e5902 45%, #162f01 100%);
            border-right: 2px solid var(--ps-border);
            overflow: hidden;
        }

        /* Grounding Platforms */
        .platform-opp {
            position: absolute;
            top: 140px;
            right: 80px;
            width: 250px;
            height: 70px;
            background: radial-gradient(ellipse, rgba(46, 89, 2, 0.9) 0%, rgba(22, 47, 1, 0.4) 60%, transparent 80%);
            border-radius: 50%;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.6);
        }
        .platform-player {
            position: absolute;
            bottom: 40px;
            left: 70px;
            width: 330px;
            height: 90px;
            background: radial-gradient(ellipse, rgba(46, 89, 2, 0.9) 0%, rgba(22, 47, 1, 0.4) 60%, transparent 80%);
            border-radius: 50%;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.6);
        }

        /* Sprites */
        .sprite-opp {
            position: absolute;
            top: 40px;
            right: 120px;
            height: 150px;
            filter: drop-shadow(0 15px 12px rgba(0,0,0,0.6));
            z-index: 2;
        }
        .sprite-player {
            position: absolute;
            bottom: 60px;
            left: 140px;
            height: 170px;
            filter: drop-shadow(0 20px 15px rgba(0,0,0,0.7));
            z-index: 2;
        }

        /* Authentic Showdown HP Cards */
        .hud-opp {
            position: absolute;
            top: 25px;
            left: 30px;
            background: var(--ps-card);
            border: 2px solid #5a769c;
            border-radius: 6px;
            padding: 8px 14px;
            width: 260px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }
        .hud-player {
            position: absolute;
            bottom: 30px;
            right: 30px;
            background: var(--ps-card);
            border: 2px solid #5a769c;
            border-radius: 6px;
            padding: 8px 14px;
            width: 280px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }

        .hud-header { display: flex; justify-content: space-between; font-weight: bold; font-size: 14px; text-shadow: 1px 1px 2px #000; }
        .lvl-badge { color: #facc15; font-size: 11px; margin-left: 6px; font-weight: normal; }
        .hp-track { width: 100%; height: 11px; background: #111a26; border: 1px solid #4a6382; border-radius: 6px; margin: 6px 0; overflow: hidden; }
        .hp-fill { height: 100%; width: 100%; background: #22c55e; transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1), background 0.5s ease; }
        .hud-sub { display: flex; justify-content: space-between; font-size: 11px; color: #cbd5e1; }

        /* Pokéball Roster Indicators */
        .ball-row { display: flex; gap: 4px; margin-top: 4px; }
        .p-ball { width: 12px; height: 12px; border-radius: 50%; background: #ef4444; border: 1px solid #ffffff; display: inline-block; }
        .p-ball.fainted { background: #475569; border-color: #64748b; opacity: 0.5; }

        /* Right-Hand Showdown Battle Log & AI Inspector */
        .side-chat {
            width: 380px;
            height: 100%;
            background: #182436;
            display: flex;
            flex-direction: column;
            border-left: 1px solid #334661;
        }
        .chat-header { background: #111a26; padding: 8px 12px; font-weight: bold; font-size: 12px; color: #73b9ff; border-bottom: 1px solid #334661; }
        .chat-log { flex: 1; padding: 10px; overflow-y: auto; font-size: 12px; line-height: 1.6; color: #e2e8f0; font-family: monospace; }
        .ai-drawer { background: #111a26; border-top: 2px solid #334661; padding: 10px; font-size: 12px; }
        .nash-row { display: flex; justify-content: space-between; background: #1c2b3e; padding: 4px 8px; border-radius: 4px; margin-top: 3px; }

        /* Bottom Controls (Moves & Switches) */
        .battle-controls {
            background: #1c293d;
            padding: 12px 20px;
            border-top: 2px solid var(--ps-border);
            display: grid;
            grid-template-columns: 2.2fr 1fr;
            gap: 16px;
        }

        /* 4 Big Showdown Move Tiles */
        .moves-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .move-button {
            border: 2px solid rgba(255,255,255,0.25);
            border-radius: 6px;
            padding: 10px 14px;
            cursor: pointer;
            text-align: left;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.4), 0 3px 6px rgba(0,0,0,0.4);
            color: #ffffff;
            text-shadow: 1px 1px 2px #000;
            transition: transform 0.1s, filter 0.1s;
        }
        .move-button:hover:not(:disabled) { transform: translateY(-2px); filter: brightness(1.15); }
        .move-button:disabled { opacity: 0.4; cursor: not-allowed; }

        /* Showdown Official Type Colors */
        .type-Normal { background: linear-gradient(#A8A878, #8A8A59); border-color: #6D6D4E; }
        .type-Fire { background: linear-gradient(#F08030, #DD6610); border-color: #B44A00; }
        .type-Water { background: linear-gradient(#6890F0, #386CEB); border-color: #1E50C5; }
        .type-Electric { background: linear-gradient(#F8D030, #E0B408); border-color: #B89000; color: #000; text-shadow: none; }
        .type-Grass { background: linear-gradient(#78C850, #5CA935); border-color: #4A892B; }
        .type-Ice { background: linear-gradient(#98D8D8, #69C6C6); border-color: #45B3B3; color: #000; text-shadow: none; }
        .type-Fighting { background: linear-gradient(#C03028, #9B221B); border-color: #78140E; }
        .type-Poison { background: linear-gradient(#A040A0, #802A80); border-color: #611861; }
        .type-Ground { background: linear-gradient(#E0C068, #D4A82F); border-color: #AA8214; color: #000; text-shadow: none; }
        .type-Flying { background: linear-gradient(#A890F0, #8662EC); border-color: #6438D9; }
        .type-Psychic { background: linear-gradient(#F85888, #F62763); border-color: #D60945; }
        .type-Bug { background: linear-gradient(#A8B820, #8D9A1B); border-color: #6D7715; }
        .type-Rock { background: linear-gradient(#B8A038, #93802D); border-color: #716223; }
        .type-Ghost { background: linear-gradient(#705898, #553E7B); border-color: #412E5E; }
        .type-Dragon { background: linear-gradient(#7038F8, #4C08F5); border-color: #3804BA; }
        .type-Steel { background: linear-gradient(#B8B8D0, #9797BA); border-color: #79799E; color: #000; text-shadow: none; }
        .type-Dark { background: linear-gradient(#705848, #513F34); border-color: #362A23; }
        .type-Fairy { background: linear-gradient(#EE99AC, #E76F87); border-color: #D34260; }

        /* Tera & Switch Column */
        .switch-col { display: flex; flex-direction: column; gap: 8px; }
        .btn-tera {
            background: linear-gradient(135deg, #c026d3, #7c3aed);
            border: 2px solid #f472b6;
            color: white;
            font-weight: bold;
            padding: 9px;
            border-radius: 6px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(192, 38, 211, 0.4);
            text-align: center;
        }
        .bench-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; }
        .bench-card {
            background: #182333;
            border: 1px solid #4a6382;
            border-radius: 4px;
            padding: 4px;
            text-align: center;
            cursor: pointer;
            transition: all 0.1s;
        }
        .bench-card:hover:not(.fainted) { background: #2d415e; border-color: #73b9ff; transform: translateY(-2px); }
        .bench-card.fainted { opacity: 0.35; cursor: not-allowed; }
        .bench-icon { width: 36px; height: 36px; display: block; margin: 0 auto; }
        .bench-hp { font-size: 10px; color: #94a3b8; font-weight: bold; }
    </style>
</head>
<body>

    <div class="showdown-client">
        <!-- Topbar -->
        <div class="topbar">
            <div class="topbar-title">POKÉMON SHOWDOWN: [Gen 9] OU Tournament Match</div>
            <div class="team-selector">
                <span style="color: #94a3b8;">Format:</span>
                <select class="select-input" id="meta-select" onchange="restartMatch()">
                    <option value="balance">Tournament Balance Core (Tusk / Gambit / Ghold)</option>
                    <option value="hyper_offense">Tournament Hyper Offense (Valiant / Moon)</option>
                    <option value="stall">Tournament Bulky Stall (Dondozo / Gliscor)</option>
                </select>
                <button class="select-input" style="cursor: pointer; background: #3b82f6; font-weight: bold;" onclick="restartMatch()">New Match</button>
            </div>
        </div>

        <!-- Main Arena + Chat -->
        <div class="main-stage">
            <div class="battle-canvas">
                <!-- Opponent Platform & Sprite -->
                <div class="platform-opp"></div>
                <img class="sprite-opp" id="opp-sprite" src="" alt="Opponent">

                <!-- Opponent HUD -->
                <div class="hud-opp">
                    <div class="hud-header">
                        <span><span id="opp-name">---</span><span class="lvl-badge">Lv. 100</span></span>
                        <span id="opp-tera-badge"></span>
                    </div>
                    <div class="hp-track"><div class="hp-fill" id="opp-hp-bar"></div></div>
                    <div class="hud-sub">
                        <div class="ball-row" id="opp-balls"></div>
                        <span id="opp-hp-num">100%</span>
                    </div>
                </div>

                <!-- Player Platform & Back Sprite -->
                <div class="platform-player"></div>
                <img class="sprite-player" id="player-sprite" src="" alt="Player">

                <!-- Player HUD -->
                <div class="hud-player">
                    <div class="hud-header">
                        <span><span id="player-name">---</span><span class="lvl-badge">Lv. 100</span></span>
                        <span id="player-tera-badge"></span>
                    </div>
                    <div class="hp-track"><div class="hp-fill" id="player-hp-bar"></div></div>
                    <div class="hud-sub">
                        <div class="ball-row" id="player-balls"></div>
                        <span id="player-hp-num">--- / --- HP</span>
                    </div>
                </div>
            </div>

            <!-- Side Chat / Inspector -->
            <div class="side-chat">
                <div class="chat-header">📜 SHOWDOWN BATTLE LOG</div>
                <div class="chat-log" id="battle-log"></div>
                <div class="ai-drawer">
                    <div style="color: #38bdf8; font-weight: bold; margin-bottom: 4px;">🧠 Glaubermon Max Nash Inspector:</div>
                    <div id="ai-nash-strat"></div>
                </div>
            </div>
        </div>

        <!-- Bottom Controls -->
        <div class="battle-controls">
            <div>
                <div style="font-size: 11px; color: #94a3b8; font-weight: bold; margin-bottom: 6px;">SELECT ATTACK MOVE:</div>
                <div class="moves-grid" id="moves-grid"></div>
            </div>

            <div class="switch-col">
                <button class="btn-tera" id="tera-btn" onclick="toggleTera()">✨ TERASTALLIZE: OFF</button>
                <div style="font-size: 11px; color: #94a3b8; font-weight: bold;">SWITCH POKÉMON:</div>
                <div class="bench-grid" id="bench-grid"></div>
            </div>
        </div>
    </div>

    <script>
        let isTera = false;

        function toggleTera() {
            isTera = !isTera;
            const btn = document.getElementById('tera-btn');
            btn.innerText = isTera ? "✨ TERA ACTIVE (READY)" : "✨ TERASTALLIZE: OFF";
            btn.style.boxShadow = isTera ? "0 0 20px #ec4899" : "none";
            btn.style.background = isTera ? "linear-gradient(135deg, #f43f5e, #a855f7)" : "linear-gradient(135deg, #c026d3, #7c3aed)";
        }

        async function fetchState() {
            const res = await fetch('/api/state');
            const data = await res.json();
            render(data);
        }

        async function restartMatch() {
            const team = document.getElementById('meta-select').value;
            await fetch('/api/restart?archetype=' + team, { method: 'POST' });
            fetchState();
        }

        function render(data) {
            // 1. Render Opponent
            if (data.ai.active) {
                document.getElementById('opp-name').innerText = data.ai.active.species;
                document.getElementById('opp-sprite').src = data.ai.active.sprite;
                const hpPct = data.ai.active.hp_percent;
                const bar = document.getElementById('opp-hp-bar');
                bar.style.width = hpPct + '%';
                bar.style.background = hpPct > 50 ? '#22c55e' : (hpPct > 20 ? '#eab308' : '#ef4444');
                document.getElementById('opp-hp-num').innerText = hpPct + '%';
                document.getElementById('opp-tera-badge').innerHTML = data.ai.active.is_tera ?
                    '<span style="background: #a855f7; padding: 2px 6px; border-radius: 4px; font-size: 10px;">TERA</span>' : '';
            }

            // Opponent Balls (6v6)
            const oppBalls = document.getElementById('opp-balls');
            oppBalls.innerHTML = '';
            data.ai.pokemon.forEach(p => {
                oppBalls.innerHTML += `<div class="p-ball ${p.is_fainted ? 'fainted' : ''}" title="${p.species}"></div>`;
            });

            // 2. Render Player
            if (data.player.active) {
                document.getElementById('player-name').innerText = data.player.active.species;
                document.getElementById('player-sprite').src = data.player.active.back_sprite;
                const hpPct = data.player.active.hp_percent;
                const bar = document.getElementById('player-hp-bar');
                bar.style.width = hpPct + '%';
                bar.style.background = hpPct > 50 ? '#22c55e' : (hpPct > 20 ? '#eab308' : '#ef4444');
                document.getElementById('player-hp-num').innerText = data.player.active.hp + ' / ' + data.player.active.max_hp + ' HP';
                document.getElementById('player-tera-badge').innerHTML = data.player.active.is_tera ?
                    '<span style="background: #a855f7; padding: 2px 6px; border-radius: 4px; font-size: 10px;">TERA</span>' : '';
            }

            // Player Balls (6v6)
            const playerBalls = document.getElementById('player-balls');
            playerBalls.innerHTML = '';
            data.player.pokemon.forEach(p => {
                playerBalls.innerHTML += `<div class="p-ball ${p.is_fainted ? 'fainted' : ''}" title="${p.species}"></div>`;
            });

            // 3. Move Buttons
            const movesGrid = document.getElementById('moves-grid');
            movesGrid.innerHTML = '';
            data.player.moves.forEach((m, idx) => {
                const btn = document.createElement('button');
                btn.className = 'move-button type-' + m.type;
                btn.innerHTML = `
                    <div style="font-weight: bold; font-size: 13px;">${m.name}</div>
                    <div style="font-size: 11px; display: flex; justify-content: space-between; margin-top: 4px;">
                        <span>${m.type.toUpperCase()} • BP ${m.bp}</span>
                        <span>PP ${m.pp}</span>
                    </div>
                `;
                btn.onclick = () => submitAction('move', idx + 1);
                movesGrid.appendChild(btn);
            });

            // 4. Bench Pokémon Switches (6v6)
            const benchGrid = document.getElementById('bench-grid');
            benchGrid.innerHTML = '';
            data.player.pokemon.forEach((p, idx) => {
                if (!p.is_active) {
                    const card = document.createElement('div');
                    card.className = 'bench-card ' + (p.is_fainted ? 'fainted' : '');
                    card.innerHTML = `
                        <img class="bench-icon" src="${p.icon}" alt="${p.species}">
                        <div class="bench-hp">${p.is_fainted ? 'FAINT' : p.hp_percent + '%'}</div>
                    `;
                    if (!p.is_fainted) {
                        card.onclick = () => submitAction('switch', idx + 1);
                    }
                    benchGrid.appendChild(card);
                }
            });

            // 5. Chat Log
            const chatLog = document.getElementById('battle-log');
            chatLog.innerHTML = data.logs.join('<br>');
            chatLog.scrollTop = chatLog.scrollHeight;

            // 6. AI Nash Analysis
            const nashBox = document.getElementById('ai-nash-strat');
            nashBox.innerHTML = `<div style="color: #94a3b8; font-size: 11px; margin-bottom: 3px;">Eval Score: <b>${data.eval_value > 0 ? '+' : ''}${data.eval_value}</b> (${data.eval_value > 0 ? 'AI Advantage' : 'Trainer Advantage'})</div>`;
            if (data.ai_analysis) {
                data.ai_analysis.forEach(s => {
                    nashBox.innerHTML += `<div class="nash-row"><span>${s.command}</span> <b>${s.prob}%</b></div>`;
                });
            }
        }

        async function submitAction(actionType, slot) {
            let body = { action_type: actionType, slot: slot, is_tera: isTera };
            isTera = false;
            document.getElementById('tera-btn').innerText = "✨ TERASTALLIZE: OFF";
            document.getElementById('tera-btn').style.boxShadow = "none";

            const res = await fetch('/api/act', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();
            render(data);
        }

        window.onload = fetchState;
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTML_TEMPLATE


@app.get("/api/state")
async def get_state():
    return serialize_battle_state(SESSION["state"])


@app.post("/api/restart")
async def restart_match(archetype: str = "balance"):
    SESSION["p1_team_name"] = archetype
    SESSION["p2_team_name"] = archetype
    SESSION["state"] = create_meta_battle(archetype, archetype)
    SESSION["logs"] = [f"<b>⚔️ New 6v6 Match Started: {archetype.upper()} Tournament Meta!</b>"]
    SESSION["last_ai_strat"] = []
    SESSION["last_eval"] = 0.0
    return {"status": "ok"}


class ActionPayload(BaseModel):
    action_type: str
    slot: int
    is_tera: bool = False


@app.post("/api/act")
async def take_action(payload: ActionPayload):
    state = SESSION["state"]
    resolver = SESSION["resolver"]

    if state.is_game_over:
        return serialize_battle_state(state)

    hu_side = state.p1
    active = hu_side.active_pokemon

    # 1. Human Action
    if payload.action_type == "move" and active:
        m = active.moves[payload.slot - 1]
        hu_act = MoveAction(move_id=m.id, move_slot=payload.slot, is_tera=payload.is_tera, tera_type=active.tera_type)
        tera_msg = " [TERA]" if payload.is_tera else ""
        SESSION["logs"].append(f"Turn {state.turn}: <b>{active.species}</b> used <b>{m.name}</b>{tera_msg}!")
    elif payload.action_type == "switch":
        target = payload.slot - 1
        hu_act = SwitchAction(target_slot=payload.slot, species=hu_side.pokemon[target].species)
        SESSION["logs"].append(f"Turn {state.turn}: Trainer sent out <b>{hu_side.pokemon[target].species}</b>!")
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

    # 2. AI Action (Depth 2 Subgame Search)
    inv_state = BattleState(p1=state.p2, p2=state.p1, weather=state.weather, terrain=state.terrain, turn=state.turn)
    ai_act, ai_strat, ai_actions, eval_val = resolver.resolve_turn(inv_state, depth=1, sample=False)

    SESSION["last_eval"] = eval_val
    SESSION["last_ai_strat"] = [
        {"command": a.to_showdown_command(), "prob": round(float(p) * 100, 1)}
        for a, p in sorted(zip(ai_actions, ai_strat), key=lambda x: x[1], reverse=True)[:5]
    ]

    ai_mon = state.p2.active_pokemon
    if ai_act.action_type == ActionType.MOVE and ai_mon:
        slot = getattr(ai_act, "move_slot") - 1
        m_name = ai_mon.moves[slot].name
        SESSION["logs"].append(f"Turn {state.turn}: Opposing <b>{ai_mon.species}</b> used <b>{m_name}</b>!")
    elif ai_act.action_type == ActionType.SWITCH:
        target = getattr(ai_act, "target_slot") - 1
        SESSION["logs"].append(f"Turn {state.turn}: Glaubermon Max sent out <b>{state.p2.pokemon[target].species}</b>!")

    # 3. Simulate turn transition
    SESSION["state"] = simulate_turn_transition(state, hu_act, ai_act)

    if SESSION["state"].is_game_over:
        winner = "Trainer (You)" if SESSION["state"].winner == 1 else "Glaubermon Max"
        SESSION["logs"].append(f"<b>🏆 MATCH FINISHED: {winner} Won!</b>")

    return serialize_battle_state(SESSION["state"])


def run_official_stadium(port: int = 8000):
    """Launch the 1:1 official styled Showdown Stadium."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  LAUNCHING OFFICIAL-STYLE SHOWDOWN BATTLE STADIUM")
    print(f"  Target Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  URL: http://localhost:{port}")
    print("=" * 70)

    model = GlaubermonMaxNet(d_model=256, nhead=8, num_actions=14).to(device)
    ckpt = "checkpoints/glaubermon_alphazero_latest.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded AlphaZero weights from {ckpt}")

    evaluator = NeuralEvaluator(model, device)
    SESSION["resolver"] = SubgameResolver(evaluator=evaluator)
    SESSION["state"] = create_meta_battle("balance", "balance")
    SESSION["logs"].append("<b>⚔️ Welcome to the Official Showdown Tournament Stadium!</b>")

    webbrowser.open(f"http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    run_official_stadium()
