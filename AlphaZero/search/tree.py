"""MCTS 核心搜索 — PUCT 选择 + 扩展 + 回传"""
import numpy as np

from ..engine.move import ActionEncoder
from ..engine.state import GameState
from .node import MCTSNode
from .evaluator import Evaluator


class MCTS:
    """蒙特卡洛树搜索。

    每次 search() 执行 num_simulations 次模拟，
    返回根节点的走法概率分布。
    """

    def __init__(self,
                 evaluator: Evaluator,
                 num_simulations: int = 800,
                 c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25):
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, root_state: GameState,
               temperature: float = 1.0) -> np.ndarray:
        """执行完整搜索。

        Args:
            root_state: 根局面
            temperature: τ，1.0 为按访问次数比例采样，→0 为贪心

        Returns:
            np.ndarray shape (POLICY_SIZE,) — 走法概率分布
        """
        if root_state.is_terminal():
            return np.zeros(ActionEncoder.POLICY_SIZE, dtype=np.float32)

        # 创建根节点并扩展
        root = MCTSNode(prior=1.0)
        self._expand(root, root_state)

        # 添加 Dirichlet 噪声到根节点（鼓励探索）
        legal = ActionEncoder.legal_indices(root_state)
        if len(legal) > 1:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(legal)
            )
            for i, idx in enumerate(legal):
                root.children[idx].prior = (
                    (1 - self.dirichlet_epsilon) * root.children[idx].prior
                    + self.dirichlet_epsilon * noise[i]
                )

        # 执行模拟
        for _ in range(self.num_simulations):
            self._simulate(root, root_state)

        # 从访问次数计算概率分布
        policy = np.zeros(ActionEncoder.POLICY_SIZE, dtype=np.float32)
        total_visits = sum(c.visit_count for c in root.children.values())

        if temperature == 0.0:
            # 贪心：选访问最多的
            best_idx = max(root.children.keys(),
                           key=lambda k: root.children[k].visit_count)
            policy[best_idx] = 1.0
        elif total_visits > 0:
            for idx, child in root.children.items():
                # P ∝ N^(1/τ)
                policy[idx] = child.visit_count ** (1.0 / temperature)
            policy /= policy.sum()

        return policy

    def select_move(self, root_state: GameState,
                    temperature: float = 1.0) -> tuple[object, np.ndarray]:
        """搜索并采样一步走法。

        Returns:
            (selected_move: Move, search_probs: np.ndarray)
        """
        probs = self.search(root_state, temperature)
        legal = ActionEncoder.legal_indices(root_state)
        if len(legal) == 0:
            return None, probs

        if temperature == 0.0:
            idx = np.argmax(probs)
        else:
            # 按概率采样
            probs_legal = probs[legal]
            probs_legal /= probs_legal.sum()  # 重新归一化
            idx = legal[np.random.choice(len(legal), p=probs_legal)]

        return ActionEncoder.decode(int(idx)), probs

    # ── 内部方法 ──

    def _simulate(self, root: MCTSNode, root_state: GameState) -> None:
        """一次完整的 MCTS 模拟：选择 → 扩展 → 评估 → 回传。"""
        node = root
        state = root_state
        path = []  # [(node, move_index), ...]

        # 1) Select: 沿 PUCT 最大路径走到叶节点
        while node.is_expanded() and not state.is_terminal():
            idx = node.select_child(self.c_puct)
            if idx < 0 or idx not in node.children:
                break
            node = node.children[idx]
            move = ActionEncoder.decode(idx)
            state = state.apply(move)
            path.append((node, idx))

        # 2) Expand + Evaluate
        if state.is_terminal():
            value = state.result() or 0.0
        else:
            # 扩展当前节点
            self._expand(node, state)
            policy, value = self.evaluator.evaluate(state)

        # 视角翻转：value 从当前方视角翻转到根方视角
        # 如果 state.turn != root_state.turn，翻转 value
        if state.turn != root_state.turn:
            value = -value

        # 3) Backup: 沿路径回传
        # 注意：path 中的节点对应的是走棋后的状态
        # value 需要沿路径交替翻转
        for n, _ in reversed(path):
            n.visit_count += 1
            n.total_value += value
            value = -value  # 下一层是对方视角

        root.visit_count += 1

    def _expand(self, node: MCTSNode, state: GameState) -> None:
        """扩展节点的所有合法子节点。"""
        policy, _ = self.evaluator.evaluate(state)
        legal = ActionEncoder.legal_indices(state)

        if len(legal) == 0:
            return

        for idx in legal:
            node.children[idx] = MCTSNode(prior=float(max(policy[idx], 0.001)))
