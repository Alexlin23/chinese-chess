# 中国象棋项目 — 全面代码审查报告

> 审查日期：2026-06-19  
> 范围：`backend/` `frontend/` `test/` 共 7 个文件

---

## 项目概览

| 文件 | 行数 | 职责 |
|------|------|------|
| `backend/chess_rules.py` | 345 | 规则引擎：走法生成、将军检测、胜负判定 |
| `backend/main.py` | 175 | FastAPI 入口：7个端点 + 静态文件托管 |
| `backend/database.py` | 144 | SQLite 持久化：对局、步数记录 |
| `backend/models.py` | 51 | Pydantic 请求/响应模型 |
| `frontend/chess.js` | 729 | 前端核心：棋盘渲染、交互、轮询同步 |
| `frontend/style.css` | 202 | 样式 |
| `test/test_api.py` | 71 | API 冒烟测试 |

---

## 一、严重问题（2 个）

### 1.1 前端本地规则缺少将军校验

**文件** `frontend/chess.js:570-591`

`getValidMovesLocal` 只返回棋子原始走法，未过滤"走后己方被将"的非法走法。后端不可用时，玩家走本地模式可以做自杀棋——主动送帅给对方吃。

```javascript
// 当前：只过滤了不能吃己方棋子（在 movesXxx 内部）
// 缺少：模拟走棋 → 检测己方是否被将
function getValidMovesLocal(row, col) {
  // 第 570 行起 — 直接返回 raw moves
}
```

**影响**：`useBackend = false` 时（后端挂了自动切本地），玩家可以走违法棋。

**修复思路**：在 `getValidMovesLocal` 中复用后端的模式——对每个候选走法原地模拟 → `isInCheckLocal(board, color)` → 撤销，只保留走后己方不将军的。

---

### 1.2 悔棋未同步到数据库

**文件** `frontend/chess.js:480-498`

悔棋只动了前端内存（`history.pop()`、恢复 `board[row][col]`），没有调后端 API 删除 `moves` 表最后一条记录，也没有更新 `games` 表的 `board_state` 和 `current_turn`。

**后果**：

- 刷新页面 → 从 DB 恢复 → 棋盘回到悔棋前的状态（数据不一致）
- `syncedStepCount` 未更新 → 轮询可能产生奇怪的跳步行为

**修复思路**：新增后端 `POST /api/game/{id}/undo` 端点：
1. 删除 `moves` 表最后一条记录
2. 把 `games` 表回退到上一步的 `board_state` + `current_turn`（或直接从 `moves` 表找上一记录）
3. 前端 `btn-undo` 调此 API，成功后更新本地状态

---

## 二、中等问题（3 个）

### 2.1 `_is_in_check` 缺少将军源剪枝

**文件** `backend/chess_rules.py:290-333`

当前对棋盘上每个敌方棋子都调 `get_raw_moves` + 遍历全部走法，无论该棋子是否真的能威胁到帅的位置。

```python
# 第 315-322 行
for r in range(ROWS):
    for c in range(COLS):
        p = board[r][c]
        if p and p["color"] == enemy_color:
            raw = get_raw_moves(board, r, c)  # 全量计算走法
            for m in raw:
                if m["row"] == kr and m["col"] == kc:
                    return True
```

**优化空间**：先根据棋子类型和帅的相对位置快速判断是否可能将军：

```python
def _can_threaten(piece_type, r, c, kr, kc):
    dr, dc = abs(r - kr), abs(c - kc)
    if piece_type in ("車",): return r == kr or c == kc
    if piece_type in ("馬",): return (dr, dc) in [(2,1),(1,2)]
    if piece_type in ("砲","炮"): return r == kr or c == kc
    if piece_type in ("卒","兵"): return dr <= 1 and dc <= 1
    if piece_type in ("將","帥"): return c == kc  # 对脸
    # 象、士 实际不可能将到帅（被九宫限制），但在将军检测中仍需保留以防 bug
    return True  # 无法快速排除，走完整计算
```

> 注意：剪枝逻辑需要仔细验证，避免漏判（比如炮翻山后暴露的战线）。

---

### 2.2 `make_move` 中深拷贝无必要

**文件** `backend/chess_rules.py:109-117`

`get_valid_moves` 已经改成原地模拟+撤销了，但 `make_move` 返回新棋盘时仍然深拷贝 90 格。

```python
def make_move(board, from_pos, to_pos):
    new_board = _copy_board(board)  # 第 113 行 — 90 个 dict 创建
    captured = new_board[tr][tc]
    new_board[tr][tc] = new_board[fr][fc]
    new_board[fr][fc] = None
    return new_board, captured
```

由于只复制了顶层 list 而棋子 dict 是共享引用（`dict(cell)`），实际上并非完全深拷贝但也不必要——走一步只改两个格子，整板复制浪费。

**优化**：用 list comprehension 浅拷贝行 + 只替换变动的行，或直接做 `[row[:] for row in board]` 即可（因为 `board[row][col] = ...` 只需两处赋值）。

---

### 2.3 前端本地兜底无将军提示

**文件** `frontend/chess.js:452-475`

`doMove` 的本地兜底分支没有将军检测，走了能将军的棋也不会显示"将军！"。

```javascript
// 第 450-475 行：本地兜底路径完全没有 isInCheck 调用
board[toR][toC] = board[fromR][fromC];
board[fromR][fromC] = null;
// 缺少：check = isInCheckLocal(board, nextTurn) 类似后端逻辑
```

**修复思路**：在 JS 端补一个 `isInCheckLocal(board, color)` 函数，本地走棋后用其检测并设置消息。

---

## 三、低优先级建议（3 个）

### 3.1 未实现长将/长捉判负和和棋

**文件** `backend/chess_rules.py:120-148`

`check_game_result` 返回值注释里写了 `"draw"` 但未实现。与 AlphaZero 自对弈训练相关：

- **长将**：同一方连续将军 ≥3 次，判负
- **长捉**：同一方连续捉同一个子 ≥3 次，判负
- **和棋**：双方同意、无子可胜等

训练模型时缺少长将判负，模型可能学会无限循环将军。

---

### 3.2 `get_raw_moves` 中 `capture` 字段在 `get_valid_moves` 内被"废弃"

**文件** `backend/chess_rules.py:66` 和 `:82-83`

`get_raw_moves` 在每个走法上设了 `m["capture"]`（第66行），但 `get_valid_moves` 做原地模拟时又用自己的局部变量 `captured`（第83行）来判断吃子——`m["capture"]` 只在最终返回的 `valid` 列表中保留给前端用。

这不是 Bug，但逻辑上有重复劳动的嫌疑。可以考虑在 `get_valid_moves` 中复用 `m["capture"]` 判断是否需要原地恢复。

---

### 3.3 `test.html` 硬编码了 `game_id` 参数但未传入

**文件** `test/test.html:161-162`

测试面板的 `doMove()` 调用 `/api/move` 时没有传 `game_id`，所以测试面板中的走棋不会写入数据库，也不会被前端轮询同步。

```javascript
const data = await apiCall("/api/move", {
  from_pos, to_pos, board: currentBoard, turn: currentTurn
  // 缺少 game_id: currentGameId
});
```

---

## 四、架构评价

### 优点

1. **规则引擎干净**：`chess_rules.py` 纯函数设计，无副作用，可直接被 AlphaZero 模块只读导入
2. **前后端分离清晰**：API 契约明确，前端本地规则兜底保证了离线可用性
3. **轮询同步机制**：0.5s 间隔 + `step_count` 比对，简洁有效
4. **增量 DOM 更新**：上一轮优化引入的 `pieceEls` 缓存，避免全量重建
5. **事务性写入**：`record_step` 用 `BEGIN IMMEDIATE` 保证棋盘+步数原子性

### 需要关注的点

1. **前端状态双源**：`board` 既来自用户点击本地更新，又来自轮询 DB 覆盖——竞态条件存在（虽然实际不太触发，因为轮询只在检测到外部变化时更新）
2. **无并发锁**：多个 test_api.py 同时跑可能竞争同一个 game_id 的步数
3. **棋规完整性**：对 AlphaZero 训练而言，需要补上长将判负规则

---

## 五、修复优先级建议

| 优先级 | 问题 | 理由 |
|--------|------|------|
| P0 | 1.2 悔棋不同步DB | 刷新后数据丢失，用户体验差 |
| P1 | 1.1 本地规则无将军校验 | 只在离线时触发，但仍是规则缺陷 |
| P1 | 2.3 本地兜底无将军提示 | 同上 |
| P2 | 2.1 将军源剪枝 | 性能优化，当前 90 格遍历开销可接受 |
| P3 | 2.2 make_move 深拷贝 | 每局只调一次走棋，影响微乎其微 |
| P3 | 3.1 长将判负 | AlphaZero 训练前必须实现 |

---

## 六、各文件逐行要点

### backend/chess_rules.py

| 行号 | 评价 |
|------|------|
| 8-17 | 初始布局用 `[col, row, type, color]` 格式，与棋盘 `[row][col]` 索引不一致，容易混淆。建议注释说明 |
| 22 | `[[None]*COLS for _ in range(ROWS)]` — 正确，没有用 `[[None]*COLS]*ROWS` 的经典陷阱 |
| 80-91 | `get_valid_moves` 原地模拟+撤销 — 上轮优化成果，正确 |
| 160-176 | `_moves_rook` — 用 `for i in range(1,10)` 代替 while，清晰 |
| 179-191 | `_moves_knight` — 马腿逻辑正确（之前已修过） |
| 194-203 | `_moves_elephant` — 象眼正确，`dr//2` 在 Python 整数除法正确 |
| 273-277 | `_copy_board` — 仍用 `dict(cell)` 深拷贝，可简化为直接 `row[:]` 浅拷贝 |
| 290-333 | `_is_in_check` — 见问题 2.1 |

### backend/main.py

| 行号 | 评价 |
|------|------|
| 66 | `is_valid_move` 做完整校验，但 `api_move` 随后又调 `make_move`，做了两次合法性校验（`is_valid_move` 内部已调 `get_valid_moves`） |
| 70-71 | `next_turn` 翻转逻辑正确 |
| 82 | `get_step_count(game_id) + 1` — 每次走棋多一次 DB 查询 |

### frontend/chess.js

| 行号 | 评价 |
|------|------|
| 253-308 | `renderPieces` 增量更新 — 上轮优化，第一遍移除、第二遍更新/创建 |
| 411 | "走法不合法" 消息没有自动清除 |
| 444 | `game_over` 覆盖了之前吃子/将军消息 — 正确（终局更重要） |
| 480-498 | 悔棋 — 见问题 1.2 |
| 511-553 | 轮询 — 正确，`data.step_count !== syncedStepCount` 触发刷新 |
| 570-591 | `getValidMovesLocal` — 见问题 1.1 |
| 562 | `sameColor` 函数定义了但从未被调用 — 死代码 |
| 677-698 | `movesCannon` 本地版 — 炮翻山后吃了子没 `break` 的问题？实际是正确的因为翻山后只能吃第一个 |

### backend/database.py

| 行号 | 评价 |
|------|------|
| 118-132 | `record_step` — `BEGIN IMMEDIATE` + 异常回滚，原子性正确 |
| 135-143 | `get_step_count` — 用 `MAX(step)` 而非 `COUNT(*)`，在有删除操作的场景下会出问题（见问题 1.2：如果悔棋删除了 moves 行，MAX(step) 会降回来，但 COUNT(*) 不会） |
