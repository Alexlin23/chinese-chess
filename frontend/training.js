"use strict";
const $ = id => document.getElementById(id);
const cards = new Map();
const number = value => Number(value || 0).toLocaleString("zh-CN");
const names = {1:"帥",2:"仕",3:"相",4:"馬",5:"車",6:"炮",7:"兵",
  "-1":"將","-2":"士","-3":"象","-4":"馬","-5":"車","-6":"砲","-7":"卒"};
const phases = {starting:"正在启动",collecting:"等待新增完整对局",training:"持续训练与自对弈",stopping:"正在保存并停止",failed:"训练异常",stopped:"已停止"};
let controlToken, stopping=false, starting=false, resumable=false, connected=false, terminal=false, pendingAction=null;
function updateStopButton(){
  $("stopTraining").disabled=!connected||!controlToken||stopping||starting||Boolean(pendingAction)||terminal;
  $("stopTraining").textContent=terminal?"训练已结束":stopping?"正在停止…":starting?"正在恢复…":resumable?"继续训练":"停止训练";
}
$("stopTraining").addEventListener("click",async()=>{
  if($("stopTraining").disabled)return;
  const action=resumable?"resume":"stop";
  pendingAction=action;stopping=action==="stop";starting=action==="resume";updateStopButton();
  $("stopStatus").textContent=starting?"正在从最后存档恢复训练…":"正在停止对弈，保存后退出训练进程…";
  try{
    const response=await fetch("/api/"+action,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({token:controlToken})});
    if(!response.ok)throw new Error("请求未被接受");
  }catch(error){
    pendingAction=null;stopping=false;starting=false;updateStopButton();
    $("stopStatus").textContent=action==="stop"?"未确认停止，请检查连接后重试。":"未能恢复训练，请查看错误信息或重试。";
  }
});
let performanceKey;
const perfLabels={"mcts.hand_wall":"MCTS每手","mcts.local_cpu_estimate_wall":"MCTS本地墙钟估算","mcts.neural_sync_wall":"MCTS同步神经调用","communication.rtt_wall":"请求往返（含推理）","inference.batch_wait_wall":"推理等待合批","inference.idle_wait_wall":"服务空闲等请求","inference.prepare_host_wall":"推理准备（主机）","inference.execution_wall":"推理执行与回传（墙钟）","inference.service_wall":"每个实际批次服务","inference.compute_cuda":"推理CUDA执行抽样","inference.compute_cuda_cold":"推理CUDA冷图抽样","inference.prepare_cuda":"推理准备CUDA区间抽样","inference.graph_warmup_wall":"CUDA图首次准备与执行","inference.batch_size":"推理实际输入Batch","inference.executed_batch_size":"推理执行Batch（含填充）","sgd.sample_wall":"Replay采样","sgd.transfer_host_wall":"SGD传输准备（主机）","sgd.compute_wall":"SGD计算及参数更新","sgd.update_wall":"SGD完整更新","sgd.transfer_cuda":"SGD传输CUDA区间抽样","sgd.compute_cuda":"SGD计算CUDA区间抽样","data.replay_write_wall":"Replay写入","data.checkpoint_wall":"Checkpoint保存","data.publish_wall":"模型导出发布"};
function renderPerformance(perf){
  const key=perf?.enabled?perf.elapsed_seconds??"waiting":"off";
  if(key===performanceKey)return;performanceKey=key;
  $("perfStatus").textContent=!perf?.enabled?"未开启 · --performance启用":perf.elapsed_seconds==null?"采集中":"每"+number(perf.interval_seconds||5)+"秒汇总 · 本次运行累计平均";
  $("perfDetails").hidden=!perf?.enabled;
  $("perfGrid").hidden=!perf?.enabled;$("perfNote").hidden=!perf?.enabled;
  if(!perf?.enabled)return;
  const m=perf.metrics||{},r=perf.rates||{};
  const ms=v=>v==null?"—":v>=1?v.toFixed(2)+" s":(v*1000).toFixed(v<.01?2:1)+" ms";
  const mean=k=>ms(m[k]?.mean),numeric=v=>v==null?"—":Number(v).toLocaleString("zh-CN",{maximumFractionDigits:1});
  $("perfMcts").textContent=mean("mcts.hand_wall");
  $("perfMctsParts").textContent="本地估算 "+mean("mcts.local_cpu_estimate_wall")+" · 同步神经调用 "+mean("mcts.neural_sync_wall");
  $("perfSimulations").textContent=numeric(r.mcts_simulations_per_second)+" 模拟/秒（完成整手）";
  $("perfInference").textContent=numeric(r.inference_positions_per_second)+" 局面/秒";
  $("perfBatch").textContent="平均输入 / 执行 Batch "+numeric(m["inference.batch_size"]?.mean)+" / "+numeric(m["inference.executed_batch_size"]?.mean);
  $("perfService").textContent="服务 "+mean("inference.service_wall")+" · 合批等待 "+mean("inference.batch_wait_wall");
  $("perfInferenceGpu").textContent="CUDA执行抽样 "+mean("inference.compute_cuda")+"（n="+number(m["inference.compute_cuda"]?.count)+"，冷图另计）";
  $("perfRtt").textContent=mean("communication.rtt_wall");$("perfRttCount").textContent=number(m["communication.rtt_wall"]?.count)+" 个请求";
  $("perfSgd").textContent=mean("sgd.update_wall");
  $("perfSgdParts").textContent="采样 "+mean("sgd.sample_wall")+" · 传输准备 "+mean("sgd.transfer_host_wall")+" · 计算更新 "+mean("sgd.compute_wall");
  $("perfSgdGpu").textContent="CUDA传输 / 计算抽样 "+mean("sgd.transfer_cuda")+" / "+mean("sgd.compute_cuda")+"（n="+number(m["sgd.compute_cuda"]?.count)+"，含首次预热）";
  $("perfCheckpoint").textContent="Checkpoint "+mean("data.checkpoint_wall");
  $("perfStorage").textContent="Replay写入 "+mean("data.replay_write_wall")+" · 模型发布 "+mean("data.publish_wall");
  $("perfPositions").textContent=numeric(r.new_positions_per_hour)+" 局面/小时";
  $("perfGames").textContent=numeric(r.completed_games_per_hour)+" 完整对局/小时";
  $("perfElapsed").textContent="已观测 "+numeric(perf.elapsed_seconds)+" 秒 · "+number(perf.sources)+" 个统计来源";
  const rows=document.createDocumentFragment();
  for(const [name,stat] of Object.entries(m)){
    const batch=name.endsWith("batch_size"),row=document.createElement("tr");
    for(const value of [perfLabels[name]||name,number(stat.count),batch?numeric(stat.mean):ms(stat.mean),batch?numeric(stat.total):ms(stat.total),batch?numeric(stat.p95_upper_approx):ms(stat.p95_upper_approx)]){
      const cell=document.createElement("td");cell.textContent=value;row.append(cell);
    }rows.append(row);
  }$("perfRows").replaceChildren(rows);
}
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
  if((pendingAction==="stop"&&(state.stop_requested||state.resume_available||state.terminal))||
     (pendingAction==="resume"&&(state.resume_requested||state.stop_available||state.errors?.length)))pendingAction=null;
  controlToken=state.stop_available||state.resume_available?state.control_token:null;
  resumable=Boolean(state.resume_available);
  stopping=Boolean(state.stop_requested)||pendingAction==="stop";
  starting=Boolean(state.resume_requested)||pendingAction==="resume";terminal=Boolean(state.terminal);
  updateStopButton();
  $("stopStatus").textContent=stopping?(terminal?(state.saved?"已保存模型 "+number(state.actor_step)+"，训练服务正在退出。":"停止过程异常，请查看错误信息。") : "正在保存并退出；未完成的对局不入库。") : starting?"正在从最后存档恢复训练…":resumable?(state.saved?"已保存模型 "+number(state.actor_step)+"，训练进程已退出。":"训练已退出，可从最后存档继续。") : "";
  renderPerformance(state.performance);
  const games=state.boards||{}, keys=Object.keys(games).sort((a,b)=>Number(a)-Number(b));
  $("active").textContent=number(keys.length);$("workers").textContent=number(state.workers)+" 盘并发 · "+number(state.actor_processes||state.workers)+" 个工作进程";
  $("games").textContent=number(state.completed_games);$("positions").textContent=number(state.positions)+" 个回放局面";
  $("step").textContent=number(state.step);$("actor").textContent="已发布模型 · "+number(state.actor_step);
  $("loss").textContent=state.loss==null?"—":state.loss.toFixed(4);
  if(state.value_loss!=null)$("lossDetail").textContent="策略 "+state.policy_loss.toFixed(3)+" / 价值 "+state.value_loss.toFixed(3);
  $("phase").textContent=phases[state.phase]||state.phase;
  $("rule").textContent="国规2020开发版 · 长捉裁定尚未完成完整验收";
  const h=state.hardware||{};
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
source.onopen=()=>{connected=true;updateStopButton();$("connection").textContent="● 实时连接";$("connection").className="connection";};
source.onerror=()=>{connected=false;updateStopButton();$("connection").textContent=stopping?"停止中 · 等待服务退出":"连接中断 · 自动重连";$("connection").className="connection offline";};
source.onmessage=event=>{try{const state=JSON.parse(event.data);render(state);
  if(state.terminal){source.close();$("connection").textContent=state.phase==="failed"?"训练异常":"训练已结束";$("connection").className="connection";}
}catch(error){console.error("Invalid training event",error);}};
$("density").addEventListener("change",event=>document.documentElement.style.setProperty("--board-min",
  ({compact:"250px",normal:"320px",large:"440px"})[event.target.value]));
