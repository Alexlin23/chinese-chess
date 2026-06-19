// ============================================================
//  中国象棋 - 前端（纯展示 + 后端API交互）
// ============================================================

const COLS = 9;
const ROWS = 10;
const CELL = 60;
const PAD  = 50;
const BOARD_W = (COLS - 1) * CELL + PAD * 2;
const BOARD_H = (ROWS - 1) * CELL + PAD * 2;

// ============================================================
//  游戏状态
// ============================================================
let board = createEmptyBoard();
let selected = null;          // {row, col}
let validMoves = [];          // [{row, col, capture}]
let turn = "r";               // "r" 红方, "b" 黑方
let gameOver = false;
let history = [];
let gameSessionId = null;
let useBackend = true;        // 是否使用后端 API
let capturedRed = [];         // 被吃的红方棋子
let capturedBlack = [];       // 被吃的黑方棋子
let syncedStepCount = 0;      // 已同步的步数，用于轮询检测外部变更
let pollingTimer = null;      // 轮询定时器
let pieceEls = Array.from({length: ROWS}, () => new Array(COLS).fill(null));  // 棋子DOM缓存
let moveIndicatorEls = [];    // 可走位置指示器DOM缓存

// DOM
const canvas  = document.getElementById("chess-board");
const ctx     = canvas.getContext("2d");
const piecesC = document.getElementById("chess-pieces");
const turnEl  = document.getElementById("current-turn");
const msgEl   = document.getElementById("status-message");

canvas.width  = BOARD_W;
canvas.height = BOARD_H;

// ============================================================
//  初始化
// ============================================================
function createEmptyBoard() {
  const b = [];
  for (let r = 0; r < ROWS; r++) b[r] = new Array(COLS).fill(null);
  return b;
}

function initBoard() {
  board = createEmptyBoard();
  // 黑方 (上方, row 0-4)
  const blacks = [
    [0,0,"車","b"],[8,0,"車","b"],
    [1,0,"馬","b"],[7,0,"馬","b"],
    [2,0,"象","b"],[6,0,"象","b"],
    [3,0,"士","b"],[5,0,"士","b"],
    [4,0,"將","b"],
    [1,2,"砲","b"],[7,2,"砲","b"],
    [0,3,"卒","b"],[2,3,"卒","b"],[4,3,"卒","b"],[6,3,"卒","b"],[8,3,"卒","b"],
  ];
  // 红方 (下方, row 5-9)
  const reds = [
    [0,9,"車","r"],[8,9,"車","r"],
    [1,9,"馬","r"],[7,9,"馬","r"],
    [2,9,"相","r"],[6,9,"相","r"],
    [3,9,"仕","r"],[5,9,"仕","r"],
    [4,9,"帥","r"],
    [1,7,"炮","r"],[7,7,"炮","r"],
    [0,6,"兵","r"],[2,6,"兵","r"],[4,6,"兵","r"],[6,6,"兵","r"],[8,6,"兵","r"],
  ];
  [...blacks, ...reds].forEach(([c, r, type, color]) => {
    board[r][c] = { type, color };
  });
}

function initGame() {
  selected = null;
  validMoves = [];
  turn = "r";
  history = [];
  gameOver = false;
  capturedRed = [];
  capturedBlack = [];
  // 清除旧棋子DOM缓存
  for (let r = 0; r < ROWS; r++)
    for (let c = 0; c < COLS; c++)
      if (pieceEls[r][c]) { pieceEls[r][c].remove(); pieceEls[r][c] = null; }
  moveIndicatorEls.forEach(el => el.remove());
  moveIndicatorEls = [];
  msgEl.textContent = "";
  updateTurn();
  renderCaptured();

  const modeEl = document.getElementById("mode-indicator");
  if (!useBackend) {
    initBoard();
    drawBoard();
    renderPieces();
    modeEl.textContent = "模式：本地规则";
    modeEl.style.color = "#ff9800";
    return;
  }

  // 尝试恢复上次对局
  const savedId = localStorage.getItem("chess_game_id");
  if (savedId) {
    fetch("/api/game/" + savedId)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => {
        if (data.status !== "ongoing") {
          // 对局已结束，创建新局
          localStorage.removeItem("chess_game_id");
          createNewGame();
          return;
        }
        // 恢复对局
        gameSessionId = data.game_id;
        board = data.board;
        turn = data.turn;
        // 恢复历史
        fetch("/api/game/" + gameSessionId + "/history")
          .then(r => r.json())
          .then(h => {
            history = h.moves.map(m => ({
              from: { row: m.from_row, col: m.from_col },
              to: { row: m.to_row, col: m.to_col },
              piece: { type: m.piece_type, color: m.piece_color },
              captured: m.captured_type ? { type: m.captured_type, color: m.captured_color } : null,
            }));
            // 恢复被吃棋子
            history.forEach(step => {
              if (step.captured) {
                if (step.captured.color === "r") capturedRed.push(step.captured);
                else capturedBlack.push(step.captured);
              }
            });
            syncedStepCount = history.length;
            startPolling();
            renderCaptured();
          });
        drawBoard();
        renderPieces();
        updateTurn();
        modeEl.textContent = "模式：后端API（已恢复对局 #" + gameSessionId + "）";
        modeEl.style.color = "#4caf50";
      })
      .catch(() => {
        // 对局不存在或加载失败，创建新局
        localStorage.removeItem("chess_game_id");
        createNewGame();
      });
  } else {
    createNewGame();
  }

  function createNewGame() {
    fetch("/api/game/new", { method: "POST" })
      .then(r => r.json())
      .then(data => {
        gameSessionId = data.game_id;
        localStorage.setItem("chess_game_id", gameSessionId);
        board = data.board;
        drawBoard();
        renderPieces();
        syncedStepCount = 0;
        startPolling();
        modeEl.textContent = "模式：后端API（对局 #" + gameSessionId + "）";
        modeEl.style.color = "#4caf50";
      })
      .catch(() => {
        useBackend = false;
        initBoard();
        drawBoard();
        renderPieces();
        modeEl.textContent = "模式：本地规则（后端未连接）";
        modeEl.style.color = "#ff9800";
      });
  }
}

// ============================================================
//  绘制棋盘
// ============================================================
function drawBoard() {
  ctx.clearRect(0, 0, BOARD_W, BOARD_H);
  ctx.fillStyle = "#f5deb3";
  ctx.fillRect(0, 0, BOARD_W, BOARD_H);

  ctx.strokeStyle = "#5d3a1a";
  ctx.lineWidth = 1;

  // 横线
  for (let r = 0; r < ROWS; r++) {
    ctx.beginPath();
    ctx.moveTo(PAD, PAD + r * CELL);
    ctx.lineTo(PAD + 8 * CELL, PAD + r * CELL);
    ctx.stroke();
  }
  // 竖线 (中间有河界断开)
  for (let c = 0; c < COLS; c++) {
    if (c === 0 || c === 8) {
      ctx.beginPath();
      ctx.moveTo(PAD + c * CELL, PAD);
      ctx.lineTo(PAD + c * CELL, PAD + 9 * CELL);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.moveTo(PAD + c * CELL, PAD);
      ctx.lineTo(PAD + c * CELL, PAD + 4 * CELL);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(PAD + c * CELL, PAD + 5 * CELL);
      ctx.lineTo(PAD + c * CELL, PAD + 9 * CELL);
      ctx.stroke();
    }
  }

  // 九宫格斜线
  ctx.lineWidth = 1;
  // 上方九宫 (黑方)
  ctx.beginPath(); ctx.moveTo(PAD + 3 * CELL, PAD); ctx.lineTo(PAD + 5 * CELL, PAD + 2 * CELL); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(PAD + 5 * CELL, PAD); ctx.lineTo(PAD + 3 * CELL, PAD + 2 * CELL); ctx.stroke();
  // 下方九宫 (红方)
  ctx.beginPath(); ctx.moveTo(PAD + 3 * CELL, PAD + 7 * CELL); ctx.lineTo(PAD + 5 * CELL, PAD + 9 * CELL); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(PAD + 5 * CELL, PAD + 7 * CELL); ctx.lineTo(PAD + 3 * CELL, PAD + 9 * CELL); ctx.stroke();

  // 楚河汉界
  ctx.fillStyle = "#5d3a1a";
  ctx.font = "bold 26px 'KaiTi', 'Microsoft YaHei', serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("楚 河", PAD + 2 * CELL, PAD + 4.5 * CELL);
  ctx.fillText("汉 界", PAD + 6 * CELL, PAD + 4.5 * CELL);

  // 行列标签 - 仅顶部列号和左侧行号
  ctx.font = "bold 13px 'Microsoft YaHei', sans-serif";
  ctx.fillStyle = "#5d3a1a";
  ctx.textBaseline = "middle";
  // 顶部列号 (0-8)
  ctx.textAlign = "center";
  for (let c = 0; c < COLS; c++) {
    ctx.fillText(String(c), PAD + c * CELL, 14);
  }
  // 左侧行号 (0-9)
  for (let r = 0; r < ROWS; r++) {
    ctx.fillText(String(r), 14, PAD + r * CELL);
  }
}

// ============================================================
//  渲染棋子（增量更新，不复建DOM）
// ============================================================
function renderPieces() {
  const captureTargets = new Set(validMoves.filter(m => m.capture).map(m => `${m.row},${m.col}`));

  // 第一遍：移除不再有棋子的位置上的DOM元素
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const el = pieceEls[r][c];
      if (el && !board[r][c]) {
        el.remove();
        pieceEls[r][c] = null;
      }
    }
  }

  // 第二遍：更新/创建棋子元素
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const p = board[r][c];
      if (!p) continue;

      let el = pieceEls[r][c];
      if (!el) {
        el = document.createElement("div");
        el.addEventListener("click", () => onPieceClick(r, c));
        piecesC.appendChild(el);
        pieceEls[r][c] = el;
      }

      const classes = `piece ${p.color === "r" ? "red" : "black"}`;
      el.className = classes
        + (selected && selected.row === r && selected.col === c ? " selected" : "")
        + (captureTargets.has(`${r},${c}`) ? " capturable" : "");
      el.textContent = p.type;
      el.style.left = (PAD + c * CELL) + "px";
      el.style.top  = (PAD + r * CELL) + "px";
    }
  }

  // 清除旧的可走指示器
  moveIndicatorEls.forEach(el => el.remove());
  moveIndicatorEls = [];

  // 创建可走指示器
  validMoves.forEach(m => {
    const el = document.createElement("div");
    el.className = m.capture ? "capture-ring" : "move-dot";
    el.style.left = (PAD + m.col * CELL) + "px";
    el.style.top  = (PAD + m.row * CELL) + "px";
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      doMove(selected.row, selected.col, m.row, m.col).catch(console.error);
    });
    piecesC.appendChild(el);
    moveIndicatorEls.push(el);
  });
}

// ============================================================
//  渲染被吃棋子
// ============================================================
function renderCaptured() {
  const blackEl = document.getElementById("captured-black");
  const redEl = document.getElementById("captured-red");
  blackEl.innerHTML = '<span class="captured-label">黑方被吃：</span>';
  capturedBlack.forEach(p => {
    const el = document.createElement("span");
    el.className = "captured-piece black";
    el.textContent = p.type;
    blackEl.appendChild(el);
  });
  redEl.innerHTML = '<span class="captured-label">红方被吃：</span>';
  capturedRed.forEach(p => {
    const el = document.createElement("span");
    el.className = "captured-piece red";
    el.textContent = p.type;
    redEl.appendChild(el);
  });
}

// ============================================================
//  交互
// ============================================================
function onPieceClick(row, col) {
  if (gameOver) return;
  const p = board[row][col];

  // 已选中棋子 → 尝试移动
  if (selected) {
    const mv = validMoves.find(m => m.row === row && m.col === col);
    if (mv) {
      doMove(selected.row, selected.col, row, col).catch(console.error);
      return;
    }
  }

  // 选择己方棋子
  if (p && p.color === turn) {
    selected = { row, col };
    fetchValidMoves(row, col);
  } else {
    selected = null;
    validMoves = [];
    renderPieces();
  }
}

// 点击棋盘空白区域取消选择
canvas.addEventListener("click", e => {
  if (!selected || gameOver) return;
  selected = null;
  validMoves = [];
  renderPieces();
});

// ============================================================
//  获取可走位置
// ============================================================
async function fetchValidMoves(row, col) {
  if (useBackend) {
    try {
      const resp = await fetch("/api/valid-moves", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ board, row, col, turn }),
      });
      const data = await resp.json();
      validMoves = data.moves || [];
    } catch (err) {
      console.warn("后端获取走法失败，使用本地规则", err);
      validMoves = getValidMovesLocal(row, col);
    }
  } else {
    validMoves = getValidMovesLocal(row, col);
  }
  renderPieces();
}

// ============================================================
//  走棋
// ============================================================
async function doMove(fromR, fromC, toR, toC) {
  if (gameOver) return;

  if (useBackend) {
    try {
      const resp = await fetch("/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          board,
          from_pos: { row: fromR, col: fromC },
          to_pos: { row: toR, col: toC },
          turn,
          game_id: gameSessionId,
        }),
      });
      const data = await resp.json();
      if (!data.valid) {
        msgEl.textContent = "走法不合法";
        return;
      }
      // 保存历史
      const captured = board[toR][toC];
      history.push({
        from: { row: fromR, col: fromC },
        to: { row: toR, col: toC },
        piece: board[fromR][fromC],
        captured,
      });
      // 每步走棋后重置消息，只反映当前步的结果
      msgEl.textContent = "";
      if (captured) {
        if (captured.color === "r") capturedRed.push(captured);
        else capturedBlack.push(captured);
        msgEl.textContent = `吃子：${captured.type}`;
        renderCaptured();
      }
      if (data.check) {
        msgEl.textContent = msgEl.textContent ? "吃子并将军！" : "将军！";
      }
      // 用后端返回的新棋盘
      board = data.new_board;
      turn = turn === "r" ? "b" : "r";
      selected = null;
      validMoves = [];
      syncedStepCount = history.length;
      updateTurn();
      drawBoard();
      renderPieces();
      if (data.game_over) {
        gameOver = true;
        msgEl.textContent = data.game_over === "red_win" ? "红方胜！" : "黑方胜！";
      }
      return;
    } catch (err) {
      console.warn("后端走棋失败，使用本地规则", err);
    }
  }

  // 本地兜底
  const capturedLocal = board[toR][toC];
  history.push({
    from: { row: fromR, col: fromC },
    to: { row: toR, col: toC },
    piece: board[fromR][fromC],
    captured: capturedLocal,
  });
  msgEl.textContent = "";
  if (capturedLocal) {
    if (capturedLocal.color === "r") capturedRed.push(capturedLocal);
    else capturedBlack.push(capturedLocal);
    msgEl.textContent = `吃子：${capturedLocal.type}`;
    renderCaptured();
  }
  board[toR][toC] = board[fromR][fromC];
  board[fromR][fromC] = null;
  turn = turn === "r" ? "b" : "r";
  selected = null;
  validMoves = [];
  updateTurn();
  drawBoard();
  renderPieces();
}

// ============================================================
//  悔棋
// ============================================================
document.getElementById("btn-undo").addEventListener("click", async () => {
  if (history.length === 0) return;

  if (useBackend && gameSessionId) {
    try {
      const resp = await fetch(`/api/game/${gameSessionId}/undo`, { method: "POST" });
      if (!resp.ok) throw new Error("undo failed");
      const data = await resp.json();
      // 用后端返回的准确状态覆盖本地
      board = data.board;
      turn = data.turn;
      syncedStepCount = data.step_count;
      // 同步历史
      const histResp = await fetch(`/api/game/${gameSessionId}/history`);
      const histData = await histResp.json();
      history = histData.moves.map(m => ({
        from: { row: m.from_row, col: m.from_col },
        to: { row: m.to_row, col: m.to_col },
        piece: { type: m.piece_type, color: m.piece_color },
        captured: m.captured_type ? { type: m.captured_type, color: m.captured_color } : null,
      }));
      capturedRed = [];
      capturedBlack = [];
      history.forEach(step => {
        if (step.captured) {
          if (step.captured.color === "r") capturedRed.push(step.captured);
          else capturedBlack.push(step.captured);
        }
      });
    } catch (err) {
      console.warn("后端悔棋失败，使用本地规则", err);
      // 本地兜底
      const last = history.pop();
      board[last.from.row][last.from.col] = last.piece;
      board[last.to.row][last.to.col] = last.captured;
      if (last.captured) {
        if (last.captured.color === "r") capturedRed.pop();
        else capturedBlack.pop();
      }
      turn = last.piece.color;
    }
  } else {
    // 纯本地模式
    const last = history.pop();
    board[last.from.row][last.from.col] = last.piece;
    board[last.to.row][last.to.col] = last.captured;
    if (last.captured) {
      if (last.captured.color === "r") capturedRed.pop();
      else capturedBlack.pop();
    }
    turn = last.piece.color;
  }

  selected = null;
  validMoves = [];
  gameOver = false;
  updateTurn();
  drawBoard();
  renderPieces();
  renderCaptured();
  msgEl.textContent = "";
});

// 重新开始
document.getElementById("btn-restart").addEventListener("click", () => {
  if (pollingTimer) clearInterval(pollingTimer);
  localStorage.removeItem("chess_game_id");
  initGame();
});

// ============================================================
//  轮询：检测外部走棋（如 test_api.py 直接调 API）
//  单次请求 /api/game/{id}，通过 step_count 检测变化
// ============================================================
function startPolling() {
  if (pollingTimer) clearInterval(pollingTimer);
  pollingTimer = setInterval(async () => {
    if (!useBackend || !gameSessionId || gameOver) return;
    try {
      const resp = await fetch(`/api/game/${gameSessionId}`);
      const data = await resp.json();
      if (data.step_count !== syncedStepCount) {
        syncedStepCount = data.step_count;
        board = data.board;
        turn = data.turn;
        // 重新加载历史（用于悔棋和被吃棋子展示）
        const histResp = await fetch(`/api/game/${gameSessionId}/history`);
        const histData = await histResp.json();
        history = histData.moves.map(m => ({
          from: { row: m.from_row, col: m.from_col },
          to: { row: m.to_row, col: m.to_col },
          piece: { type: m.piece_type, color: m.piece_color },
          captured: m.captured_type ? { type: m.captured_type, color: m.captured_color } : null,
        }));
        capturedRed = [];
        capturedBlack = [];
        history.forEach(step => {
          if (step.captured) {
            if (step.captured.color === "r") capturedRed.push(step.captured);
            else capturedBlack.push(step.captured);
          }
        });
        selected = null;
        validMoves = [];
        msgEl.textContent = "";
        updateTurn();
        drawBoard();
        renderPieces();
        renderCaptured();
        if (data.status !== "ongoing") {
          gameOver = true;
          msgEl.textContent = data.status === "red_win" ? "红方胜！" : "黑方胜！";
        }
      }
    } catch (e) { /* 忽略轮询错误 */ }
  }, 500);
}
function updateTurn() {
  turnEl.textContent = turn === "r" ? "红方" : "黑方";
  turnEl.style.color = turn === "r" ? "#c62828" : "#1a1a1a";
}

// ============================================================
//  本地走法规则（后端不可用时的兜底）
// ============================================================
function inBounds(r, c) {
  return r >= 0 && r < ROWS && c >= 0 && c < COLS;
}

function sameColor(p1, p2) {
  return p1 && p2 && p1.color === p2.color;
}

function getValidMovesLocal(row, col) {
  const piece = board[row][col];
  if (!piece) return [];
  const t = piece.type;
  const c = piece.color;
  let moves = [];

  switch (t) {
    case "車": moves = movesRook(row, col, c); break;
    case "馬": moves = movesKnight(row, col, c); break;
    case "相":
    case "象": moves = movesElephant(row, col, c); break;
    case "仕":
    case "士": moves = movesAdvisor(row, col, c); break;
    case "帥":
    case "將": moves = movesKing(row, col, c); break;
    case "炮":
    case "砲": moves = movesCannon(row, col, c); break;
    case "兵":
    case "卒": moves = movesPawn(row, col, c); break;
  }
  return moves;
}

function movesRook(r, c, color) {
  const res = [];
  const dirs = [[0,1],[0,-1],[1,0],[-1,0]];
  for (const [dr,dc] of dirs) {
    let nr = r+dr, nc = c+dc;
    while (inBounds(nr,nc)) {
      const t = board[nr][nc];
      if (!t) { res.push({row:nr,col:nc, capture:false}); }
      else {
        if (t.color !== color) res.push({row:nr,col:nc, capture:true});
        break;
      }
      nr += dr; nc += dc;
    }
  }
  return res;
}

function movesKnight(r, c, color) {
  const res = [];
  const jumps = [
    [-2,-1,-1,0],[-2,1,-1,0],
    [2,-1,1,0],[2,1,1,0],
    [-1,-2,0,-1],[-1,2,0,1],
    [1,-2,0,-1],[1,2,0,1],
  ];
  for (const [dr,dc,lr,lc] of jumps) {
    const nr = r+dr, nc = c+dc;
    if (!inBounds(nr,nc)) continue;
    // 蹩马腿
    if (board[r+lr][c+lc]) continue;
    const t = board[nr][nc];
    if (!t || t.color !== color) res.push({row:nr,col:nc, capture: !!t});
  }
  return res;
}

function movesElephant(r, c, color) {
  const res = [];
  const jumps = [[-2,-2],[-2,2],[2,-2],[2,2]];
  for (const [dr,dc] of jumps) {
    const nr = r+dr, nc = c+dc;
    if (!inBounds(nr,nc)) continue;
    // 象眼
    if (board[r+dr/2][c+dc/2]) continue;
    // 不能过河
    if (color === "r" && nr < 5) continue;
    if (color === "b" && nr > 4) continue;
    const t = board[nr][nc];
    if (!t || t.color !== color) res.push({row:nr,col:nc, capture: !!t});
  }
  return res;
}

function movesAdvisor(r, c, color) {
  const res = [];
  const jumps = [[-1,-1],[-1,1],[1,-1],[1,1]];
  for (const [dr,dc] of jumps) {
    const nr = r+dr, nc = c+dc;
    if (!inBounds(nr,nc)) continue;
    // 九宫格限制
    if (color === "r" && (nr < 7 || nc < 3 || nc > 5)) continue;
    if (color === "b" && (nr > 2 || nc < 3 || nc > 5)) continue;
    const t = board[nr][nc];
    if (!t || t.color !== color) res.push({row:nr,col:nc, capture: !!t});
  }
  return res;
}

function movesKing(r, c, color) {
  const res = [];
  const dirs = [[0,1],[0,-1],[1,0],[-1,0]];
  for (const [dr,dc] of dirs) {
    const nr = r+dr, nc = c+dc;
    if (!inBounds(nr,nc)) continue;
    if (color === "r" && (nr < 7 || nc < 3 || nc > 5)) continue;
    if (color === "b" && (nr > 2 || nc < 3 || nc > 5)) continue;
    const t = board[nr][nc];
    if (!t || t.color !== color) res.push({row:nr,col:nc, capture: !!t});
  }
  return res;
}

function movesCannon(r, c, color) {
  const res = [];
  const dirs = [[0,1],[0,-1],[1,0],[-1,0]];
  for (const [dr,dc] of dirs) {
    let nr = r+dr, nc = c+dc;
    let jumped = false;
    while (inBounds(nr,nc)) {
      const t = board[nr][nc];
      if (!jumped) {
        if (!t) res.push({row:nr,col:nc, capture:false});
        else jumped = true;
      } else {
        if (t) {
          if (t.color !== color) res.push({row:nr,col:nc, capture:true});
          break;
        }
      }
      nr += dr; nc += dc;
    }
  }
  return res;
}

function movesPawn(r, c, color) {
  const res = [];
  const forward = color === "r" ? -1 : 1;
  // 过河判断
  const crossed = color === "r" ? r <= 4 : r >= 5;

  // 向前
  const nr = r + forward;
  if (inBounds(nr, c)) {
    const t = board[nr][c];
    if (!t || t.color !== color) res.push({row:nr, col:c, capture: !!t});
  }
  // 过河后可左右
  if (crossed) {
    for (const dc of [-1, 1]) {
      const nc = c + dc;
      if (inBounds(r, nc)) {
        const t = board[r][nc];
        if (!t || t.color !== color) res.push({row:r, col:nc, capture: !!t});
      }
    }
  }
  return res;
}

// ============================================================
//  启动
// ============================================================
initGame();
