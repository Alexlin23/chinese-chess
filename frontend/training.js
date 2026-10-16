"use strict";
const $ = id => document.getElementById(id);
const cards = new Map();
const number = value => Number(value || 0).toLocaleString("zh-CN");
const names = {1:"帥",2:"仕",3:"相",4:"馬",5:"車",6:"炮",7:"兵",
  "-1":"將","-2":"士","-3":"象","-4":"馬","-5":"車","-6":"砲","-7":"卒"};
const phases = {starting:"正在启动",collecting:"收集首批完整对局",training:"持续训练与自对弈",failed:"训练异常",stopped:"已停止"};
function draw(canvas, game) {
  const ctx = canvas.getContext("2d"), scale = 38, pad = 28, width=360, height=398;
  canvas.width=720; canvas.height=796; ctx.scale(2,2);
  ctx.fillStyle="#decba7";ctx.fillRect(0,0,width,height);
  const x=c=>pad+c*scale,y=r=>pad+r*scale;
  ctx.strokeStyle="#826e4e";ctx.lineWidth=.7;
  for(let r=0;r<10;r++){ctx.beginPath();ctx.moveTo(x(0),y(r));ctx.lineTo(x(8),y(r));ctx.stroke();}
  for(let c=0;c<9;c++){
    ctx.beginPath();ctx.moveTo(x(c),y(0));ctx.lineTo(x(c),y(c===0||c===8?9:4));ctx.stroke();
    if(c!==0&&c!==8){ctx.beginPath();ctx.moveTo(x(c),y(5));ctx.lineTo(x(c),y(9));ctx.stroke();}
  }
  for(const start of [0,7]){ctx.beginPath();ctx.moveTo(x(3),y(start));ctx.lineTo(x(5),y(start+2));ctx.moveTo(x(5),y(start));ctx.lineTo(x(3),y(start+2));ctx.stroke();}
  ctx.fillStyle="#826e4e";ctx.font='17px "KaiTi","STKaiti",serif';ctx.textAlign="center";ctx.textBaseline="middle";
  ctx.fillText("楚 河",x(2),y(4.5));ctx.fillText("漢 界",x(6),y(4.5));
  if(game.last_move){const [r,c,tr,tc]=game.last_move;ctx.strokeStyle="#58836a";ctx.lineWidth=2.8;ctx.beginPath();ctx.arc(x(c),y(r),14,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(x(tc),y(tr),17,0,Math.PI*2);ctx.stroke();}
  for(let r=0;r<10;r++)for(let c=0;c<9;c++){
    const piece=game.board[r][c];if(!piece)continue;
    ctx.shadowColor="#72563055";ctx.shadowBlur=3;ctx.shadowOffsetY=2;
    ctx.beginPath();ctx.arc(x(c),y(r),15.3,0,Math.PI*2);ctx.fillStyle="#f3e6c9";ctx.fill();
    ctx.shadowBlur=0;ctx.shadowOffsetY=0;ctx.lineWidth=1.2;ctx.strokeStyle=piece>0?"#a04435":"#384b43";ctx.stroke();
    ctx.beginPath();ctx.arc(x(c),y(r),12.7,0,Math.PI*2);ctx.lineWidth=.5;ctx.stroke();
    ctx.fillStyle=piece>0?"#a3352c":"#25372f";ctx.font='bold 21px "KaiTi","STKaiti","Microsoft YaHei",serif';ctx.fillText(names[piece],x(c),y(r)+.7);
  }
}
function makeCard(worker){
  const card=document.createElement("article");card.className="game";
  const head=document.createElement("div");head.className="game-head";
  const title=document.createElement("span");title.className="game-title";
  const turn=document.createElement("span");turn.className="turn";head.append(title,turn);
  const canvas=document.createElement("canvas");canvas.setAttribute("role","img");
  const foot=document.createElement("div");foot.className="game-foot";
  const detail=document.createElement("span"),model=document.createElement("span");foot.append(detail,model);
  card.append(head,canvas,foot);$("boards").append(card);
  return {card,title,turn,canvas,detail,model,key:""};
}
function render(state){
  const games=state.boards||{}, keys=Object.keys(games).sort((a,b)=>Number(a)-Number(b));
  $("active").textContent=number(keys.length);$("workers").textContent=number(state.workers)+" 盘并发 · "+number(state.actor_processes||state.workers)+" 个工作进程";
  $("games").textContent=number(state.completed_games);$("positions").textContent=number(state.positions)+" 个回放局面";
  $("step").textContent=number(state.step);$("actor").textContent="已发布模型 · "+number(state.actor_step);
  $("loss").textContent=state.loss==null?"—":state.loss.toFixed(4);
  if(state.value_loss!=null)$("lossDetail").textContent="策略 "+state.policy_loss.toFixed(3)+" / 价值 "+state.value_loss.toFixed(3);
  $("phase").textContent=phases[state.phase]||state.phase;
  $("rule").textContent="国规2020开发版 · 长捉裁定尚未完成完整验收";
  const h=state.hardware||{};$("gpu").textContent=(h.gpu||"GPU")+" · "+(h.gpu_percent??"—")+"%";
  $("vram").textContent="显存 · "+(h.vram_gb??0).toFixed(1)+" / "+(h.vram_total_gb??0).toFixed(1)+" GB";
  $("cpu").textContent="CPU · "+(h.cpu_percent??"—")+"%";$("ram").textContent=(h.memory_scope||"系统")+"内存 · "+(h.ram_gb??0).toFixed(1)+" / "+(h.ram_total_gb??0).toFixed(1)+" GB";
  $("empty").hidden=keys.length>0;
  for(const [worker,card] of cards)if(!Object.hasOwn(games,worker)){card.card.remove();cards.delete(worker);}
  for(const worker of keys){
    const game=games[worker];let view=cards.get(worker);
    if(!view){view=makeCard(worker);cards.set(worker,view);}
    const slot=keys.indexOf(worker),at=$("boards").children[slot];
    if(at!==view.card)$("boards").insertBefore(view.card,at||null);
    view.title.textContent="对局 "+(Number(worker)+1)+" · 第 "+(game.sequence+1)+" 局";
    view.turn.textContent=game.turn===1?"红方行棋":"黑方行棋";view.turn.className="turn"+(game.turn===1?" red":"");
    view.detail.textContent="第 "+game.ply+" 手 · "+Math.max(0,Math.floor(Date.now()/1000-game.started))+" 秒";
    view.model.textContent=game.ply===0?"首次搜索中":"模型 "+number(game.actor_step)+" · 上一手 "+number(game.diagnostics?.simulations||0)+" 次搜索";
    const key=game.episode+":"+game.ply;
    if(view.key!==key){draw(view.canvas,game);view.canvas.setAttribute("aria-label",view.title.textContent+"，第 "+game.ply+" 手，"+view.turn.textContent);view.key=key;}
  }
  const errors=state.errors||[];$("errors").hidden=!errors.length;$("errors").textContent=errors.map(e=>e.error).join("\n");
}
const source=new EventSource("/events");
source.onopen=()=>{$("connection").textContent="● 实时连接";$("connection").className="connection";};
source.onerror=()=>{$("connection").textContent="连接中断 · 自动重连";$("connection").className="connection offline";};
source.onmessage=event=>{try{const state=JSON.parse(event.data);render(state);
  if(state.terminal){source.close();$("connection").textContent=state.phase==="failed"?"训练异常":"训练已结束";$("connection").className="connection";}
}catch(error){console.error("Invalid training event",error);}};
$("density").addEventListener("change",event=>document.documentElement.style.setProperty("--board-min",
  ({compact:"250px",normal:"320px",large:"440px"})[event.target.value]));
