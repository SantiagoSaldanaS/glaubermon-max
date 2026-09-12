"""Paired Gen 9 OU pilot using the official simulator and unmodified poke-env policy.

Run from repo root. No accounts, websocket connection, or replay uploads are used.
Both adapters consume only the player's own official protocol channel and request.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from poke_env.battle import Battle
from poke_env.player import SimpleHeuristicsPlayer

from glaubermon.client.showdown_bot import ShowdownBot, PACKED_TEAMS


from glaubermon.evaluation.showdown_transport import OfficialBridge, CapturedSocket, choice_text


async def play_game(args, game_id, mode, out):
    block, swap = divmod(game_id, 2)
    keys = ['balance', 'stall', 'pelol']  # hyper_offense fails official OU validation (Roaring Moon)
    pairs = [(a, b) for a in keys for b in keys if a != b]
    team_names = list(pairs[block % len(pairs)])
    # Fixed team at each official side; agents swap sides and therefore teams.
    bot_side = swap
    seed = [block + 1, args.seed, 23456, 34567]
    random.seed(args.seed + game_id)
    np.random.seed(args.seed + game_id)
    torch.manual_seed(args.seed + game_id)
    bot = ShowdownBot(username='Glaubermon', depth=args.depth, team=team_names[bot_side],
                      checkpoint=args.checkpoint, stealth=False, evaluator=mode, load_config=False)
    forward_count = [0]
    def count_forward(*unused):
        forward_count[0] += 1
    hook = bot.model.register_forward_hook(count_forward)
    socket = bot.ws = CapturedSocket()
    baseline = SimpleHeuristicsPlayer(battle_format='gen9ou', start_listening=False,
                                      log_level=logging.ERROR)
    control = Battle('battle-gen9ou-local', 'Control', logging.getLogger('control'), gen=9)
    bridge = OfficialBridge(args.showdown)
    names = ['Control', 'Control']
    names[bot_side] = 'Glaubermon'
    start = time.monotonic()
    stats = dict(id=game_id, block=block, mode=mode, bot_side=bot_side, teams=team_names,
                 seed=seed, decisions=0, invalid_actions=[], fallbacks=0, latencies=[],
                 status='running', winner=None)
    public = []
    trace_path = out / f'{mode}-{game_id:03d}.jsonl'
    try:
        frame = bridge.exchange(dict(op='start', teams=[PACKED_TEAMS[k] for k in team_names],
                                     names=names, seed=seed))
        with trace_path.open('w') as trace:
            for phase in range(1500):
                trace.write(json.dumps({'frame': frame}) + '\n')
                public.extend(frame['public'])
                if frame['errors']:
                    stats['invalid_actions'].extend(frame['errors'])
                    # Invalid action is an agent failure, never silently replaced.
                    failed = bot_side
                    for error in frame['errors']:
                        if isinstance(error, dict):
                            failed = error['side']
                            break
                    frame = bridge.exchange(dict(op='forfeit', side=failed))
                    stats['status'] = 'invalid_action_forfeit'
                    continue
                if frame['ended']:
                    stats['winner'] = frame['winner']
                    if stats['status'] == 'running':
                        stats['status'] = 'completed'
                    break
                if frame['turn'] >= args.max_turns:
                    stats['status'] = 'truncated'
                    break
                choices = [None, None]
                for side, channel in enumerate(frame['players']):
                    lines = channel['lines']
                    req = channel['request']
                    if side == bot_side:
                        # Requests are awaited explicitly, never scheduled as racing tasks.
                        safe_lines = [l for l in lines if not l.startswith(('|request|', '|win|', '|tie|'))]
                        await bot.handle_message('>battle-gen9ou-local\n' + '\n'.join(safe_lines))
                        if req and not req.get('wait'):
                            socket.messages.clear()
                            t0 = time.monotonic()
                            await asyncio.wait_for(bot.handle_battle_turn('battle-gen9ou-local', req),
                                                   timeout=args.decision_seconds)
                            stats['latencies'].append(time.monotonic() - t0)
                            stats['decisions'] += 1
                            messages = [m for m in socket.messages if '|/choose ' in m or '|/team ' in m]
                            if len(messages) != 1:
                                raise RuntimeError(f'Expected one choice, got {messages}')
                            choices[side] = choice_text(messages[0])
                            stats['fallbacks'] += choices[side] == 'default'
                    else:
                        for line in lines:
                            if line.startswith('|') and line not in ('|', ''):
                                parts = line.split('|')
                                if parts[1] not in ('win', 'tie', 'request'):
                                    control.parse_message(parts)
                        if req and not req.get('wait'):
                            control.parse_request(req)
                            order = baseline.teampreview(control) if req.get('teamPreview') else baseline.choose_move(control)
                            choices[side] = choice_text(str(order))
                trace.write(json.dumps({'choices': choices}) + '\n')
                trace.flush()
                if all(c is None for c in choices):
                    raise RuntimeError('No pending choice and no terminal result')
                frame = bridge.exchange(dict(op='choose', choices=choices))
            else:
                stats['status'] = 'phase_limit'
        stats['turns'] = frame['turn']
    except asyncio.TimeoutError:
        stats['status'] = 'decision_timeout_forfeit'
        stats['winner'] = 'Control'
    except Exception as exc:
        stats['status'] = 'infrastructure_error'
        stats['error'] = repr(exc)
    finally:
        bridge.close()
        hook.remove()
        stats['neural_forward_calls'] = forward_count[0]
        stats['seconds'] = time.monotonic() - start
        (out / f'{mode}-{game_id:03d}.log').write_text('\n'.join(public) + '\n')
        (out / f'{mode}-{game_id:03d}.json').write_text(json.dumps(stats, indent=2))
    return stats


def run_job(args, game, mode, out):
    torch.set_num_threads(1)
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger('glaubermon').setLevel(logging.ERROR)
    return asyncio.run(play_game(args, game, mode, out))


async def main(args):
    if args.games < 2 or args.games % 2:
        raise ValueError("Paired benchmark requires a positive even number of games")
    if Path('showdown_config.json').exists():
        raise RuntimeError('Run in isolated checkout without showdown_config.json')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger('glaubermon').setLevel(logging.ERROR)
    manifest = dict(vars(args), python=platform.python_version(), hardware=platform.platform(),
                    torch=torch.__version__, poke_env=importlib.metadata.version('poke-env'),
                    source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest(),
                    checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                    teams={k:PACKED_TEAMS[k] for k in ['balance','stall','pelol']},
                    bridge_sha256=hashlib.sha256(Path(__file__).with_name('showdown_bridge.cjs').read_bytes()).hexdigest(),
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    showdown_version=json.loads((Path(args.showdown)/'package.json').read_text())['version'],
                    poke_env_source=importlib.metadata.distribution('poke-env').read_text('direct_url.json'),
                    rules='gen9ou', transport='official Battle with player-filtered protocol IPC',
                    decision_timeout_policy='bot loses', invalid_action_policy='offending side loses',
                    turn_cap_policy='truncated, no fabricated winner', threads=1)
    mf = out / 'manifest.json'
    if mf.exists() and json.loads(mf.read_text()) != manifest:
        raise ValueError('Output manifest differs; use a new directory')
    mf.write_text(json.dumps(manifest, indent=2))
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [loop.run_in_executor(pool, run_job, args, game, mode, out)
                for mode in args.modes for game in range(args.games)
                if not (out / f'{mode}-{game:03d}.json').exists()]
        for job in asyncio.as_completed(jobs):
            result = await job
            print(json.dumps({k:v for k,v in result.items() if k != 'latencies'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--showdown', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint', default='checkpoints/glaubermon_rebel_latest.pt')
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--modes', nargs='+', choices=['hybrid','heuristic'], default=['hybrid','heuristic'])
    p.add_argument('--seed', type=int, default=911)
    p.add_argument('--max-turns', type=int, default=300)
    p.add_argument('--decision-seconds', type=float, default=30)
    asyncio.run(main(p.parse_args()))
