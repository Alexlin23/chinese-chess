"use strict";
const $=id=>document.getElementById(id);
const names = {1:"帥",2:"仕",3:"相",4:"馬",5:"車",6:"炮",7:"兵",
  "-1":"將","-2":"士","-3":"象","-4":"馬","-5":"車","-6":"砲","-7":"卒"};
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

let state=null,selected=null,busy=false,requestEpoch=0;
function screen(r,c){return state.human===-1?[9-r,8-c]:[r,c];}
function render(next){
  const changed=!state||next.game_id!==state.game_id||next.revision!==state.revision;
  state=next;if(changed)selected=null;
  let board=state.board,last=state.last_move;
  if(state.human===-1){board=board.slice().reverse().map(row=>row.slice().reverse());if(last)last=[9-last[0],8-last[1],9-last[2],8-last[3]];}
  const canvas=$("board");draw(canvas,{board,last_move:last});
  if(selected){
    const ctx=canvas.getContext("2d"),[r,c]=screen(...selected);
    ctx.strokeStyle="#218653";ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(28+c*38,28+r*38,17,0,Math.PI*2);ctx.stroke();
    for(const move of state.legal_moves.filter(m=>m[0]===selected[0]&&m[1]===selected[1])){
      const [tr,tc]=screen(move[2],move[3]);ctx.fillStyle="#218653";ctx.beginPath();ctx.arc(28+tc*38,28+tr*38,4,0,Math.PI*2);ctx.fill();
    }
  }
  $("model").textContent="已训练 "+state.model_step.toLocaleString("zh-CN")+" 步 · 每手 "+state.simulations+" 次搜索";
  $("ply").textContent="第 "+state.ply+" 手";
  $("error").textContent=state.error||"";
  $("status").textContent=state.result?(state.result.winner===null?"和棋":state.result.winner===state.human?"你赢了":"模型获胜"):state.error?"模型运行异常":state.thinking?"模型正在思考…":state.turn===state.human?"轮到你走棋":"等待模型";
}
async function refresh(){
  if(busy)return;
  const epoch=++requestEpoch;
  try{const response=await fetch("/api/state");if(!response.ok)throw Error("读取棋局失败");const next=await response.json();if(epoch===requestEpoch)render(next);}
  catch(error){$("error").textContent="连接中断，正在重试："+error.message;}
}
async function post(path,data){
  if(busy||!state)return;
  busy=true;++requestEpoch;
  try{const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...data,token:state.token})});const next=await response.json();if(!response.ok)throw Error(next.error||"操作失败");render(next);}
  catch(error){$("error").textContent=error.message;}
  finally{busy=false;}
}
$("board").addEventListener("click",event=>{
  if(!state||busy||state.thinking||state.result||state.error||state.turn!==state.human)return;
  const box=event.currentTarget.getBoundingClientRect();
  let c=Math.round(((event.clientX-box.left)/box.width*360-28)/38),r=Math.round(((event.clientY-box.top)/box.height*398-28)/38);
  if(r<0||r>9||c<0||c>8)return;
  [r,c]=screen(r,c);
  if(selected){const move=[...selected,r,c];if(state.legal_moves.some(m=>m.every((v,i)=>v===move[i]))){post("/api/move",{game_id:state.game_id,revision:state.revision,move});return;}}
  selected=state.board[r][c]*state.human>0?[r,c]:null;render(state);
});
$("restart").addEventListener("click",()=>post("/api/new",{human:Number($("human").value)}));
refresh();setInterval(refresh,500);
