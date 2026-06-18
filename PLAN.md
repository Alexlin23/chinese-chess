# 中国象棋 - 实现规划

## 项目结构

```
中国象棋/
├── frontend/
│   ├── index.html          # 主页面
│   ├── style.css           # 样式
│   └── chess.js            # 前端逻辑（棋盘绘制、交互、临时规则）
├── backend/
│   ├── main.py             # FastAPI 服务入口 & 路由
│   ├── chess_rules.py      # 棋子走法规则引擎
│   ├── models.py           # Pydantic 数据模型
│   ├── database.py         # SQLite 数据库操作
│   └── requirements.txt    # Python 依赖
└── PLAN.md                 # 本文件
```

## 功能模块

### 1. 棋子可选位置检测 (`chess_rules.py`)
- `get_valid_moves(board, row, col, turn)` → 返回合法目标位置列表
- 每种棋子独立走法逻辑：車、馬、象/相、士/仕、將/帥、砲/炮、卒/兵
- 包含：蹩马腿、塞象眼、将帅对面、不能送将 等限制

### 2. 棋子移动方法 (`chess_rules.py`)
- `make_move(board, from_pos, to_pos)` → 返回新棋盘状态
- `is_valid_move(board, from_pos, to_pos, turn)` → 校验走法合法性
- 走棋后自动检测是否被将军

### 3. 胜负判定 (`chess_rules.py`)
- `check_game_result(board, turn)` → 返回 "red_win" / "black_win" / "ongoing" / "draw"
- 判胜：吃掉将/帅
- 判和：无子可动（困毙）、长将/长捉（简化版可选）

### 4. API 接口 (`main.py`)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/valid-moves` | 获取某棋子可走位置 |
| POST | `/api/move` | 执行走棋，返回结果 |
| POST | `/api/check-win` | 检测胜负 |
| POST | `/api/game/new` | 创建新对局 |
| GET  | `/api/game/{id}` | 获取对局信息 |
| GET  | `/api/game/{id}/history` | 获取走棋历史 |

### 5. 数据库存储 (`database.py`)
- 表 `games`：对局ID、创建时间、当前状态、当前回合
- 表 `moves`：走棋记录（对局ID、步数、起止位置、吃子信息）
- 使用 SQLite，轻量无需额外部署

## 技术栈

- **前端**：HTML + CSS + JavaScript（原生，无框架）
- **后端**：Python + FastAPI
- **数据库**：SQLite
- **API 协议**：REST + JSON
