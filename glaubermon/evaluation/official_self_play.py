"""Self-play trajectories from official transitions and the actual live observation path.

No truth state is available to either policy. Opponent estimates are the same public
meta priors and revealed information used by ShowdownBot. Forced replacement targets
imitate the live replacement selector; normal targets are sampled Nash policies.
"""
import asyncio
import json
from pathlib import Path

import numpy as np
from glaubermon.client.showdown_bot import ShowdownBot, PACKED_TEAMS
from glaubermon.core.actions import action_to_logit_index
from glaubermon.models.embeddings import encode_battle_state
from glaubermon.evaluation.showdown_transport import OfficialBridge, CapturedSocket, choice_text


class RecordingResolver:
    def __init__(self, resolver):
        self.resolver = resolver
        self.sample_policy = False
        self.last = None

    def resolve_turn(self, *args, **kwargs):
        args = list(args)
        if len(args) > 2:
            args[2] = self.sample_policy
        else:
            kwargs['sample'] = self.sample_policy
        self.last = self.resolver.resolve_turn(*args, **kwargs)
        return self.last


async def collect_game(model, showdown, team_names, seed, depth=1, max_turns=300,
                       trace_path=None, decision_seconds=30, team_pool=None):
    team_pool = PACKED_TEAMS if team_pool is None else team_pool
    names = ['SelfPlay1', 'SelfPlay2']
    bots = [ShowdownBot(username=names[i], team=team_names[i], depth=depth,
                       model=model, load_config=False, evaluator='hybrid', stealth=False)
            for i in range(2)]
    for bot in bots:
        bot.ws = CapturedSocket()
        bot.resolver = RecordingResolver(bot.resolver)
    bridge = OfficialBridge(showdown)
    trajectory = []
    trace = Path(trace_path).open('w') if trace_path else None
    try:
        frame = bridge.exchange(dict(op='start', names=names, seed=seed,
                                     teams=[team_pool[t] for t in team_names]))
        for phase in range(1500):
            if trace:
                trace.write(json.dumps({'frame':frame}) + '\n')
                trace.flush()
            if frame['errors']:
                raise RuntimeError(f"Official self-play rejected an action: {frame['errors']}")
            if frame['ended'] or frame['turn'] >= max_turns:
                break
            choices = [None,None]
            for i, channel in enumerate(frame['players']):
                bot = bots[i]
                lines = [l for l in channel['lines'] if not l.startswith(('|request|','|win|','|tie|'))]
                await bot.handle_message('>battle-gen9ou-selfplay\n'+'\n'.join(lines))
                req = channel['request']
                if not req or req.get('wait'):
                    continue
                is_preview = bool(req.get('teamPreview'))
                bot.resolver.sample_policy = not req.get('forceSwitch') and not is_preview
                bot.resolver.last = None
                bot.ws.messages.clear()
                # This uses the exact live adapter, with no access to the other request.
                tensors = None if is_preview else encode_battle_state(bot.build_battle_state('battle-gen9ou-selfplay',req))
                await asyncio.wait_for(bot.handle_battle_turn('battle-gen9ou-selfplay',req), decision_seconds)
                messages = [m for m in bot.ws.messages if '|/choose ' in m or '|/team ' in m]
                if len(messages) != 1:
                    raise RuntimeError(f'Expected one official self-play choice: {messages}')
                choices[i] = choice_text(messages[0])
                if is_preview:
                    continue
                policy = np.zeros(14,dtype=np.float32)
                if req.get('forceSwitch'):
                    if not choices[i].startswith('switch '):
                        raise RuntimeError(f'Invalid replacement: {choices[i]}')
                    # Logit positions 8..13 correspond to team slots 1..6.
                    policy[8 + int(choices[i].split()[1]) - 1] = 1.0
                else:
                    if bot.resolver.last is None or choices[i] == 'default':
                        raise RuntimeError('No valid search policy for training')
                    _, strategy, actions, _ = bot.resolver.last
                    for action, probability in zip(actions,strategy):
                        policy[action_to_logit_index(action)] += probability
                if not np.isfinite(policy).all() or not np.isclose(policy.sum(),1):
                    raise RuntimeError('Invalid policy distribution')
                trajectory.append((i,tensors,policy))
                if trace:
                    trace.write(json.dumps({"sample_side":i,"target_policy":policy.tolist()})+"\n")
            if trace:
                trace.write(json.dumps({'choices':choices})+'\n')
            if all(c is None for c in choices):
                raise RuntimeError('No pending choice in nonterminal official game')
            frame = bridge.exchange(dict(op='choose',choices=choices))
        # Truncation is missing outcome, never a draw or heuristic pseudo-result.
        outcomes = [None,None]
        if frame['ended']:
            outcomes = [0.0,0.0] if not frame['winner'] else [1.0 if frame['winner']==n else -1.0 for n in names]
        samples = [(tensors,policy,outcomes[i]) for i,tensors,policy in trajectory]
        info = dict(terminated=frame['ended'], winner=frame['winner'],
                    turns=frame['turn'], samples=len(samples), seed=seed, teams=team_names)
        if trace:
            trace.write(json.dumps({"result":info,"value_targets_by_side":outcomes})+"\n")
        return samples, info
    finally:
        bridge.close()
        if trace:
            trace.close()
