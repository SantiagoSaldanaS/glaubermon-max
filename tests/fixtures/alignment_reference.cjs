// Official, isolated mechanics fixtures. Gen 9 custom game permits controlled sets.
const {Battle}=require(process.argv[2]);
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
function mon(p) {return {species:p.species.name,types:p.getTypes(),hp:p.hp,maxhp:p.maxhp,stats:p.storedStats,
 ability:p.ability,item:p.item,status:p.status,time:p.statusState.time||0,boosts:p.boosts,
 moves:p.moveSlots.map(m=>({id:m.id,pp:m.pp,maxpp:m.maxpp}))};}
function snap(b) {return {sides:b.sides.map(s=>({active:s.pokemon.indexOf(s.active[0]),mons:s.pokemon.map(mon),
 screens:Object.fromEntries(['reflect','lightscreen','auroraveil'].filter(k=>s.sideConditions[k]).map(k=>[k,s.sideConditions[k].duration])),
 tailwind:s.sideConditions.tailwind?.duration||0})),trick_room:b.field.pseudoWeather.trickroom?.duration||0};}
const output=[];
for (const fixture of input) {
 const team=fixture.teams||[[{species:'Snorlax',moves:fixture.moves?.[0]||['Seismic Toss']}], [{species:'Snorlax',moves:fixture.moves?.[1]||['Splash']}]];
 const b=new Battle({formatid:'gen9customgame',seed:fixture.seed||[19,2,3,4],strictChoices:true,
 p1:{name:'P1',team:team[0].map(m=>({ability:'Thick Fat',nature:'Hardy',...m}))},
 p2:{name:'P2',team:team[1].map(m=>({ability:'Thick Fat',nature:'Hardy',...m}))}});
 try {
 b.makeChoices('team 123456'.slice(0,5+team[0].length),'team 123456'.slice(0,5+team[1].length));
 for(let i=0;i<2;i++) {
  const p=b.sides[i].active[0], c=fixture.initial?.[i]||{};
  if(c.hp!==undefined)p.hp=c.hp;
  if(c.status){p.setStatus(c.status);if(c.time!==undefined)p.statusState.time=c.time;}
  if(c.boosts)Object.assign(p.boosts,c.boosts);
  if(c.tailwind)b.sides[i].addSideCondition('tailwind',p);
  for(const k of c.screens||[])b.sides[i].addSideCondition(k,p);
 }
 if(fixture.trick_room)b.field.addPseudoWeather('trickroom',b.sides[0].active[0]);
 const before=snap(b), results=[];
 for(const choices of fixture.actions||[['move 1','move 1']]) { b.makeChoices(...choices);results.push(snap(b));if(b.ended)break; }
 output.push({name:fixture.name,before,results,log:b.log});
 } finally {b.destroy();}
}
console.log(JSON.stringify(output));
