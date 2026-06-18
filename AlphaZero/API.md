# 中国象棋 API 文档

> 后端地址：`http://127.0.0.1:8001`  
> 所有 POST 请求需 `Content-Type: application/json`

---

## 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 返回前端页面 |
| GET | `/test` | 返回 API 测试页面 |
| POST | `/api/valid-moves` | 查询棋子合法走法 |
| POST | `/api/move` | 执行走棋 |
| POST | `/api/check-win` | 检测胜负 |
| POST | `/api/game/new` | 创建新对局 |
| GET | `/api/game/{game_id}` | 获取对局信息 |
| GET | `/api/game/{game_id}/history` | 获取走棋历史 |

---

## 数据结构

### 棋子 `PieceInfo`

每个棋盘格为 `null` 或以下对象：

```json
{ "type": "馬", "color": "r" }
```

**`type` 取值**：`帥` `仕` `相` `車` `馬` `炮` `兵` （红方）  
　　　　　　`將` `士` `象` `車` `馬` `砲` `卒` （黑方）

**`color` 取值**：`"r"` 红方，`"b"` 黑方

### 棋盘 `board`

10 × 9 的二维数组，`board[row][col]`：
- `row` 0–9：0 = 顶部（黑方底线），9 = 底部（红方底线）
- `col` 0–8：0 = 左侧，8 = 右侧

### 位置 `Position`

```json
{ "row": 0, "col": 0 }
```

---

## 详细接口

### 1. 查询合法走法

```
POST /api/valid-moves
```

**请求体**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `row` | int | 棋子所在行 |
| `col` | int | 棋子所在列 |
| `board` | `(PieceInfo|null)[][]` | 当前棋盘 |
| `turn` | string | 当前轮次 `"r"` / `"b"` |

**示例**：
```json
{
  "row": 1,
  "col": 0,
  "board": [[...]],
  "turn": "b"
}
```

**响应**：
```json
{
  "moves": [
    { "row": 2, "col": 2, "capture": false },
    { "row": 3, "col": 1, "capture": true }
  ]
}
```

`moves` 数组每项：`row` 目标行、`col` 目标列、`capture` 是否吃子。

---

### 2. 执行走棋

```
POST /api/move
```

**请求体**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `from_pos` | `Position` | 起始位置 |
| `to_pos` | `Position` | 目标位置 |
| `board` | `(PieceInfo|null)[][]` | 当前棋盘 |
| `turn` | string | 当前轮次 |
| `game_id` | int \| null | 对局 ID，传入则同步写入数据库 |

**示例**：
```json
{
  "from_pos": { "row": 1, "col": 0 },
  "to_pos": { "row": 2, "col": 2 },
  "board": [[...]],
  "turn": "b",
  "game_id": 1
}
```

**响应** `MoveResponse`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `valid` | bool | 走法是否合法 |
| `captured` | `PieceInfo` \| null | 被吃的棋子（null 表示没吃子） |
| `check` | bool | 走完是否将军 |
| `game_over` | string \| null | `"red_win"` / `"black_win"` / null |
| `new_board` | `(PieceInfo|null)[][]` | 走完后的新棋盘 |

**示例**：
```json
{
  "valid": true,
  "captured": { "type": "兵", "color": "r" },
  "check": false,
  "game_over": null,
  "new_board": [[...]]
}
```

> **注意**：`valid: false` 时仅返回 `{ "valid": false }`。

---

### 3. 检测胜负

```
POST /api/check-win
```

**请求体**：同 `ValidMovesRequest`

```json
{
  "board": [[...]],
  "row": 0,
  "col": 0,
  "turn": "r"
}
```

**响应**：
```json
{
  "result": "red_win"
}
```

`result` 取值：`"red_win"` / `"black_win"` / `"ongoing"`

---

### 4. 创建新对局

```
POST /api/game/new
```

**请求体**：无

**响应**：
```json
{
  "game_id": 1,
  "board": [[...]],
  "turn": "r"
}
```

新局始终红方先行（`"r"`）。

---

### 5. 获取对局信息

```
GET /api/game/{game_id}
```

**响应**：
```json
{
  "game_id": 1,
  "board": [[...]],
  "turn": "r",
  "status": "ongoing"
}
```

`status` 取值：`"ongoing"` / `"red_win"` / `"black_win"`

---

### 6. 获取走棋历史

```
GET /api/game/{game_id}/history
```

**响应**：
```json
{
  "game_id": 1,
  "moves": [
    {
      "id": 1,
      "step": 1,
      "from_row": 7, "from_col": 1,
      "to_row": 5, "to_col": 2,
      "piece_type": "馬", "piece_color": "r",
      "captured_type": null, "captured_color": null,
      "created_at": "2026-06-19 12:00:00"
    }
  ]
}
```

---

## 前端轮询机制

前端 `chess.js` 每 **500ms** 轮询一次 `/api/game/{id}/history`，通过比对步数检测外部 API 调用（如 `test_api.py`）产生的棋盘变化，并自动刷新界面。

---

## 错误码

| 状态码 | 说明 |
|--------|------|
| 200 | 正常 |
| 404 | 对局不存在（`/api/game/{id}`） |
| 422 | 请求体格式不符（Pydantic 校验失败） |
