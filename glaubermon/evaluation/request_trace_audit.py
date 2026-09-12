"""Replay filtered official traces and compare own action menus and active PP.

This does not certify hidden opponent state or stochastic simulator transitions.
"""
import sys,asyncio,json,logging,argparse,contextlib
from pathlib import Path
import torch
from glaubermon.client.showdown_bot import ShowdownBot
from glaubermon.models.set_transformer import GlaubermonMaxNet
from glaubermon.core.types import ActionType
async def audit(path,model):
 bots=[ShowdownBot(username=name,model=model,load_config=False,stealth=False) for name in ('SelfPlay1','SelfPlay2')]
 checked=0;errors=[]
 for lineno,line in enumerate(path.read_text().splitlines(),1):
  row=json.loads(line);frame=row.get('frame')
  if not frame:continue
  for i,channel in enumerate(frame['players']):
   b=bots[i];b.username=(channel.get('request') or {}).get('side',{}).get('name',b.username)
   lines=[l for l in channel['lines'] if not l.startswith(('|request|','|win|','|tie|'))]
   # Side comes from the explicitly filtered channel, never opponent-private data.
   b.our_side['battle-trace']='p'+str(i+1)
   await b.handle_message('>battle-trace\n'+'\n'.join(lines))
   req=channel.get('request')
   if not req or req.get('wait') or req.get('teamPreview') or frame['ended']:continue
   state=b.build_battle_state('battle-trace',req);checked+=1;expected=set();actual=set()
   force=bool(req.get('forceSwitch'));active=(req.get('active') or [{}])[0]
   if not force:
    for slot,move in enumerate(active.get('moves',[]),1):
     if move.get('disabled') or move.get('pp',1)<=0:continue
     expected.add(('move',move['id'],False))
     if active.get('canTerastallize'):expected.add(('move',move['id'],True))
   if force or not active.get('trapped'):
    for slot,p in enumerate(req['side']['pokemon'],1):
     if not p.get('active') and not p['condition'].startswith(('0 ','0/')) and p['condition']!='0 fnt':expected.add(('switch',slot,False))
   for a in state.get_valid_actions(1):
    actual.add(('move',a.move_id,a.is_tera) if a.action_type==ActionType.MOVE else ('switch',a.target_slot,False))
   if expected!=actual:errors.append(dict(line=lineno,turn=frame['turn'],side=i+1,missing=sorted(expected-actual),extra=sorted(actual-expected)))
   if not force and active.get('moves') and active['moves'][0]['id']!='struggle':
    for m in active['moves']:
     local=next((x for x in state.p1.active_pokemon.moves if x.id==m['id']),None)
     if not local or local.pp!=m['pp'] or not local.pp_known:errors.append(dict(line=lineno,turn=frame['turn'],side=i+1,pp_mismatch=m['id']))
 return dict(file=str(path),requests=checked,discrepancies=errors)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('traces',type=Path,nargs='+')
 parser.add_argument('--output',type=Path)
 args=parser.parse_args()
 logging.disable(logging.CRITICAL)
 model=GlaubermonMaxNet(d_model=32,nhead=4).eval()
 with contextlib.redirect_stdout(sys.stderr):
  results=[asyncio.run(audit(path,model)) for path in args.traces]
 report=dict(purpose='Own legal menus and active PP only; no claim of full transition parity',
  requests=sum(r['requests'] for r in results),discrepancies=sum(len(r['discrepancies']) for r in results),traces=results)
 text=json.dumps(report,indent=2)+'\n'
 if args.output:args.output.write_text(text)
 else:print(text,end='')
 return int(bool(report['discrepancies']))

if __name__=='__main__':raise SystemExit(main())
