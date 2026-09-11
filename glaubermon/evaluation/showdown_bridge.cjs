// JSON-lines transport around the official simulator. Only own private channel
// and own request are returned to each policy; no omniscient Battle crosses IPC.
const root = process.argv[2];
const {Battle, Teams, TeamValidator} = require(root);
const {extractChannelMessages} = require(root + '/dist/sim/battle.js');
const readline = require('node:readline');
let battle, cursor = 0, errors = [];
function frame() {
  const lines = battle.log.slice(cursor).filter(x => !x.startsWith('|t:|')).join('\n');
  cursor = battle.log.length;
  const channels = extractChannelMessages(lines, [0, 1, 2]);
  return {turn: battle.turn, ended: battle.ended, winner: battle.winner || null,
    public: channels[0], errors: errors.splice(0),
    players: battle.sides.map((s, i) => ({lines: channels[i + 1],
      request: battle.ended || s.isChoiceDone() ? null : s.activeRequest}))};
}
(async () => {
for await (const line of readline.createInterface({input: process.stdin})) {
  try {
    const msg = JSON.parse(line);
    if (msg.op === 'start') {
      if (battle) battle.destroy();
      const teams = msg.teams.map(t => Teams.unpack(t));
      const validation = teams.map(t => new TeamValidator('gen9ou').validateTeam(t));
      if (validation.some(x => x?.length)) throw new Error(JSON.stringify(validation));
      cursor = 0; errors = [];
      battle = new Battle({formatid:'gen9ou', seed:msg.seed,
        send(type, data) { if (type === 'sideupdate' && data.includes('|error|')) errors.push(data); },
        p1:{name:msg.names[0], team:teams[0]}, p2:{name:msg.names[1], team:teams[1]}});
    } else if (msg.op === 'choose') {
      for (let i = 0; i < 2; i++) {
        if (msg.choices[i] != null && !battle.choose('p' + (i + 1), msg.choices[i])) {
          errors.push({side:i, choice:msg.choices[i], error:battle.sides[i].choice.error});
        }
      }
    } else if (msg.op === 'forfeit') {
      battle.win(battle.sides[1 - msg.side]);
    } else { throw new Error('Unknown operation'); }
    process.stdout.write(JSON.stringify(frame()) + '\n');
  } catch (e) { process.stdout.write(JSON.stringify({fatal:String(e), stack:e.stack}) + '\n'); }
}

})();
