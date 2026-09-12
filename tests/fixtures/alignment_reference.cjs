// Official, isolated mechanics fixtures. Gen 9 custom game permits controlled sets.
const {Battle,Teams}=require(process.argv[2]);
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
function mon(p) {return {species:p.species.name.replace(/-Tera$/, ''),level:p.level,types:p.getTypes(),baseTypes:[...p.baseSpecies.types],rawTypes:[...p.types],proteanUsed:!!p.abilityState.protean,fallen:p.abilityState.fallen,tera:p.terastallized,teraType:p.teraType,hp:p.hp,maxhp:p.maxhp,stats:{...p.storedStats},
 lastMove:p.lastMove?.id||null,shieldBoost:!!p.shieldBoost,swordBoost:!!p.swordBoost,ability:p.ability,item:p.item,status:p.status,time:p.statusState.time||0,boosts:{...p.boosts},
 volatiles:Object.fromEntries(['substitute','taunt','confusion','partiallytrapped','trapped','encore','disable','leechseed','protosynthesis','quarkdrive','flashfire','roost'].filter(k=>p.volatiles[k]).map(k=>{
 const v=p.volatiles[k];return [k,{...(v.bestStat?{best_stat:v.bestStat,from_booster:!!v.fromBooster}:{}),...(v.move?{move:v.move}:{}),...(k==='leechseed'?{source_side:Number(v.sourceSlot[1])}:{}),...(v.hp!==undefined?{hp:v.hp}:{}),...(v.duration!==undefined?{duration:v.duration}:{}),
 ...(v.time!==undefined?{time:v.time}:{}),...(v.boundDivisor?{divisor:v.boundDivisor}:{}),
 ...(['partiallytrapped','trapped'].includes(k)&&v.source?{source:[v.source.side.n+1,v.source.species.name]}:{})}];})),
 moves:p.moveSlots.map(m=>({id:m.id,pp:m.pp,maxpp:m.maxpp}))};}
function legal(s,b) {
 const req=s.activeRequest;
 if(b.ended||!req||req.wait)return [];
 const result=[], force=!!req.forceSwitch?.[0], menu=req.active?.[0]||{};
 if(!force) for(const m of menu.moves||[]) {
  if(m.disabled||(m.pp!==undefined&&m.pp<=0))continue;
  result.push(['move',m.id,false]);
  if(menu.canTerastallize)result.push(['move',m.id,true]);
 }
 if(force||!menu.trapped)for(const p of s.pokemon)if(!p.isActive&&!p.fainted)result.push(['switch',p.species.name.replace(/-Tera$/,''),false]);
 return result;
}
function snap(b) {return {ended:b.ended,weather:b.field.weather,weather_turns:b.field.weatherState.duration||0,terrain:b.field.terrain,terrain_turns:b.field.terrainState.duration||0,turn:b.turn,pending:b.ended?[]:b.sides.filter(s=>s.activeRequest?.forceSwitch?.[0]).map(s=>s.n+1),sides:b.sides.map(s=>({legal:legal(s,b),active:s.pokemon.indexOf(s.active[0]),mons:s.pokemon.map(mon),
 hazards:Object.fromEntries(['stealthrock','spikes','toxicspikes','stickyweb'].filter(k=>s.sideConditions[k]).map(k=>[k,s.sideConditions[k].layers||1])),
 screens:Object.fromEntries(['reflect','lightscreen','auroraveil'].filter(k=>s.sideConditions[k]).map(k=>[k,s.sideConditions[k].duration])),
 tailwind:s.sideConditions.tailwind?.duration||0})),trick_room:b.field.pseudoWeather.trickroom?.duration||0};}
const output=[];
for (const fixture of input) {
 const team=(fixture.packed_teams?fixture.packed_teams.map(t=>Teams.unpack(t)):null)||fixture.teams||[[{species:'Snorlax',moves:fixture.moves?.[0]||['Seismic Toss']}], [{species:'Snorlax',moves:fixture.moves?.[1]||['Splash']}]];
 const b=new Battle({formatid:'gen9customgame',seed:fixture.seed||[19,2,3,4],strictChoices:true,
 p1:{name:'P1',team:team[0].map(m=>({ability:'Thick Fat',nature:'Hardy',...m}))},
 p2:{name:'P2',team:team[1].map(m=>({ability:'Thick Fat',nature:'Hardy',...m}))}});
 try {
 if(fixture.controlled_draws) {
  // Semantic outcomes shared with the Python fixture, not the production RNG.
  b.random=(m,n)=>n!==undefined?m+Math.floor((n-m)/2):m!==undefined?Math.floor(m/2):0.5;
  b.randomChance=(n,d)=>0.5<n/d;
  b.sample=items=>items[0]?.fixtureIndex!==undefined?[...items].sort((a,b)=>a.fixtureIndex-b.fixtureIndex)[0]:items[0];
  b.prng.shuffle=()=>{};
 }
 for(const side of b.sides)side.pokemon.forEach((p,i)=>p.fixtureIndex=i);
 b.makeChoices('team 123456'.slice(0,5+team[0].length),'team 123456'.slice(0,5+team[1].length));
 for(let i=0;i<2;i++) {
  const p=b.sides[i].active[0], c=fixture.initial?.[i]||{};
  if(c.fallen!==undefined){p.abilityState.fallen=c.fallen;p.side.totalFainted=c.fallen;}
  if(c.last_move)p.lastMove=b.dex.getActiveMove(c.last_move);
  if(c.hp!==undefined)p.hp=c.hp;
  if(c.status){p.setStatus(c.status);if(c.time!==undefined)p.statusState.time=c.time;}
  if(c.boosts)Object.assign(p.boosts,c.boosts);
  if(c.stats)Object.assign(p.storedStats,c.stats);
  if(c.pp)p.moveSlots.forEach((m,i)=>{m.pp=c.pp[i];});
  for(const k of c.hazards||[]) b.sides[i].addSideCondition(k,p);
  for(const [key,config] of Object.entries(c.volatiles||{})) {
   p.addVolatile(key,b.sides[1-i].active[0],b.dex.moves.get(key==='partiallytrapped'?'magmastorm':key));
   Object.assign(p.volatiles[key],config);
  }
  if(c.tailwind)b.sides[i].addSideCondition('tailwind',p);
  for(const k of c.screens||[])b.sides[i].addSideCondition(k,p);
 }
 if(fixture.weather){b.field.setWeather(fixture.weather,b.sides[0].active[0]);if(fixture.weather_turns!==undefined)b.field.weatherState.duration=fixture.weather_turns;}
 if(fixture.terrain){b.field.setTerrain(fixture.terrain,b.sides[0].active[0]);if(fixture.terrain_turns!==undefined)b.field.terrainState.duration=fixture.terrain_turns;}
 if(fixture.trick_room)b.field.addPseudoWeather('trickroom',b.sides[0].active[0]);
 if(fixture.refresh_disabled)for(const side of b.sides)b.runEvent('DisableMove',side.active[0]);
 if(fixture.initial?.some(c=>c.pp)) b.makeRequest("move");
 const damageTrace=[];
 for(const side of b.sides) for(const pokemon of side.pokemon){
 const damage=pokemon.damage;
 pokemon.damage=function(amount,source,effect){
  const value=damage.call(this,amount,source,effect);
  if(typeof value==='number' && value>0)damageTrace.push({side:this.side.n+1,damage:value,effect:effect?.id||effect,critical:!!(effect && typeof effect==='object' && effect.moveHitData?.[this.getSlot()]?.crit)});
  return value;
 };
 }
 const before=snap(b), results=[], choicesUsed=[];
 const attempts=new Map();
 function autoChoice(side) {
  const req=side.activeRequest;
  if(!req||req.wait||side.isChoiceDone())return '';
  const switches=side.pokemon.map((p,i)=>({p,i})).filter(({p})=>!p.isActive&&!p.fainted);
  const active=side.active[0];
  if(req.forceSwitch?.[0])return 'switch '+(switches[0].i+1);
  const menu=req.active[0];
  if(!menu.trapped&&switches.length&&b.turn%(side.n?11:8)===0)return 'switch '+(switches[0].i+1);
  const moves=menu.moves.map((m,i)=>({m,i})).filter(({m})=>!m.disabled&&(m.pp===undefined||m.pp>0));
  const count=attempts.get(active)||0;attempts.set(active,count+1);
  const chosen=moves[count%moves.length];
  return 'move '+(chosen.i+1)+(menu.canTerastallize&&b.turn>=(side.n?6:3)?' terastallize':'');
 }
 if(fixture.auto_steps) {
  for(let i=0;i<fixture.auto_steps&&!b.ended;i++) {
   const choices=b.sides.map(autoChoice);choicesUsed.push(choices);b.makeChoices(...choices);results.push(snap(b));
  }
 } else {
  for(const choices of fixture.actions||[['move 1','move 1']]) {choicesUsed.push(choices);b.makeChoices(...choices);results.push(snap(b));if(b.ended)break;}
 }
 output.push({name:fixture.name,before,results,choices:choicesUsed,ended:b.ended,winner:b.winner,damageTrace,log:b.log});
 } catch(error) {throw new Error(fixture.name+": "+error.message,{cause:error});} finally {b.destroy();}
}
console.log(JSON.stringify(output));
