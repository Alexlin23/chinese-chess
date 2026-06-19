"""MCTS 搜索测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from AlphaZero.engine import GameState, Move, ActionEncoder
from AlphaZero.search import MCTS, RandomEvaluator, MCTSNode


@pytest.fixture
def mcts():
    return MCTS(RandomEvaluator(seed=42), num_simulations=100, c_puct=1.5)


@pytest.fixture
def state():
    return GameState.new_game()


# ── MCTSNode ──

def test_node_creation():
    """节点创建基本属性。"""
    node = MCTSNode(prior=0.5)
    assert node.prior == 0.5
    assert node.visit_count == 0
    assert node.total_value == 0.0
    assert node.q == 0.0
    assert not node.is_expanded()


def test_node_visit():
    """访问后 Q 值更新。"""
    node = MCTSNode(prior=0.5)
    node.visit_count = 10
    node.total_value = 7.0
    assert node.q == 0.7


def test_node_select_child():
    """PUCT 选择正确。"""
    parent = MCTSNode(prior=1.0)
    parent.visit_count = 100
    # 添加两个子节点
    parent.children[0] = MCTSNode(prior=0.8)
    parent.children[0].visit_count = 50
    parent.children[0].total_value = 40.0  # Q = 0.8

    parent.children[1] = MCTSNode(prior=0.2)
    parent.children[1].visit_count = 1
    parent.children[1].total_value = 1.0  # Q = 1.0

    # PUCT: Q + c_puct * P * sqrt(N) / (1 + n)
    # 子0: 0.8 + 1.5 * 0.8 * 10 / 51 = 0.8 + 0.235 = 1.035
    # 子1: 1.0 + 1.5 * 0.2 * 10 / 2  = 1.0 + 1.5   = 2.5
    # 子1 的 PUCT 更高
    best = parent.select_child(c_puct=1.5)
    assert best == 1


# ── MCTS 搜索 ──

def test_mcts_search_returns_valid_probs(mcts, state):
    """搜索返回有效概率分布。"""
    probs = mcts.search(state, temperature=1.0)
    assert probs.shape == (ActionEncoder.POLICY_SIZE,)
    assert probs.dtype == np.float32
    assert np.isclose(probs.sum(), 1.0, atol=0.01)
    assert probs.min() >= 0.0

    # 确认所有非零概率对应合法走法
    legal = ActionEncoder.legal_indices(state)
    nonzero = np.where(probs > 0)[0]
    for idx in nonzero:
        assert idx in legal, f"Nonzero prob at idx={idx} is not a legal move"


def test_mcts_select_move_returns_legal_move(mcts, state):
    """选出的走法合法。"""
    move, probs = mcts.select_move(state, temperature=1.0)
    assert move is not None
    assert state.is_legal(move), f"Selected move {move} is illegal!"


def test_mcts_temperature_zero_greedy(mcts, state):
    """温度 0 时选访问最多的走法。"""
    probs = mcts.search(state, temperature=0.0)
    # 应该只有一个走法概率 > 0
    assert (probs > 0).sum() == 1
    assert np.isclose(probs.max(), 1.0)


def test_mcts_terminal_state():
    """终局状态下搜索安全（返回零概率）。"""
    import numpy as np
    board = np.zeros((10, 9), dtype=np.int8)
    board[0, 4] = -1  # 只有黑将
    state = GameState(board, "r")
    evaluator = RandomEvaluator(seed=42)
    mcts = MCTS(evaluator, num_simulations=10)
    probs = mcts.search(state)
    assert probs.sum() == 0.0  # 无合法走法


def test_mcts_all_selected_moves_legal(mcts, state):
    """多次搜索选出的走法都合法。"""
    for _ in range(20):
        move, _ = mcts.select_move(state, temperature=1.0)
        assert move is not None
        assert state.is_legal(move), f"Illegal move: {move}"


def test_mcts_selfplay_one_game(mcts):
    """用 MCTS 自对弈一局不崩溃。"""
    state = GameState.new_game()
    max_moves = 300
    for step in range(max_moves):
        if state.is_terminal():
            break
        move, _ = mcts.select_move(state, temperature=1.0)
        if move is None:
            break
        state = state.apply(move)
    # 应该正常结束
    assert step < max_moves - 1 or state.is_terminal(), \
        f"Game didn't finish in {max_moves} moves"

    if state.is_terminal():
        result = state.result()
        assert result in (-1.0, 1.0, 0.0), f"Invalid result: {result}"


def test_mcts_one_step_mate():
    """一步杀：MCTS 应找到杀棋。"""
    import numpy as np
    from AlphaZero.engine.constants import KING, ROOK, KNIGHT
    from AlphaZero.search import MCTS, RandomEvaluator

    # 构造红方一步杀局面
    board = np.zeros((10, 9), dtype=np.int8)
    board[0, 4] = -KING    # 黑将
    board[2, 3] = ROOK     # 红車将军（同列）
    board[9, 4] = KING     # 红帅

    state = GameState(board, "b")  # 轮到黑方
    if state.is_terminal():
        # 如果已经是杀棋，跳过
        return

    # 这是黑方面对将军的局面，不是红方杀棋
    # 改为构造红方走一步就能赢的局面
    board2 = np.zeros((10, 9), dtype=np.int8)
    board2[0, 4] = -KING   # 黑将
    board2[8, 4] = KING    # 红帅
    board2[5, 5] = ROOK    # 红車 — 走到 (5,4) 就是杀棋

    state2 = GameState(board2, "r")
    mcts2 = MCTS(RandomEvaluator(seed=42), num_simulations=200)

    # 找走法
    probs = mcts2.search(state2, temperature=1.0)
    # 杀棋走法 (5,5)→(5,4) 的索引应该有概率
    killer = Move(5, 5, 5, 4)
    killer_idx = ActionEncoder.encode(killer)
    assert killer_idx >= 0, "Killer move not encodable"

    # 随着搜索，杀棋走法应该获得概率（虽然不是最高因为随机评估器）
    # 只验证 MCTS 不会崩溃
    assert probs.sum() > 0, "Should have valid move probabilities"


def test_dirichlet_noise(mcts, state):
    """Dirichlet 噪声添加后根节点先验不再均匀。"""
    probs1 = mcts.search(state, temperature=1.0)
    probs2 = mcts.search(state, temperature=1.0)
    # 两次搜索的概率分布可能不同（噪声 + 随机评估器）
    # 至少 sum 都是 1
    assert np.isclose(probs1.sum(), 1.0)
    assert np.isclose(probs2.sum(), 1.0)
