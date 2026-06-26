"""MCTS 核心搜索 — Leaf-Parallel Batch 版本

关键优化:
  1. 串行选择 batch_size 个叶节点
  2. 收集后一次 batch evaluate（GPU batch 推理）
  3. 批量回传结果
"""
import numpy as np

from ..engine.move import ActionEncoder
from ..engine.state import GameState
from ..engine.constants import POLICY_SIZE
from .node import MCTSNode
from .evaluator import Evaluator


class MCTS:
    """Leaf-Parallel Batch MCTS。

    每轮:
      1. 串行选择 batch_size 个叶节点
      2. batch evaluate 所有叶节点
      3. 批量回传
    """

    def __init__(self,
                 evaluator: Evaluator,
                 num_simulations: int = 800,
                 c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 batch_size: int = 32,
                 virtual_loss: float = 3.0):
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.virtual_loss = virtual_loss

    def search(self, root_state: GameState,
               temperature: float = 1.0) -> np.ndarray:
        """执行完整搜索，返回 (8100,) 策略分布。"""
        if root_state.is_terminal():
            return np.zeros(POLICY_SIZE, dtype=np.float32)

        # 创建根节点并扩展
        root = MCTSNode(prior=1.0)
        self._expand(root, root_state)

        # 添加 Dirichlet 噪声到根节点
        legal_moves = root_state.legal_moves()
        if len(legal_moves) > 1:
            legal_actions = ActionEncoder.legal_actions(legal_moves)
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(legal_actions)
            )
            for i, idx in enumerate(legal_actions):
                if idx in root.children:
                    root.children[idx].prior = (
                        (1 - self.dirichlet_epsilon) * root.children[idx].prior
                        + self.dirichlet_epsilon * noise[i]
                    )

        # ── Batch 搜索 ──
        sims_done = 0
        while sims_done < self.num_simulations:
            batch_count = min(self.batch_size, self.num_simulations - sims_done)

            # 1) 串行选择 batch_count 个叶节点
            leaves = []
            for _ in range(batch_count):
                leaf = self._select_leaf(root, root_state)
                if leaf is not None:
                    leaves.append(leaf)

            if not leaves:
                break

            # 2) 分离终局和非终局
            terminal_leaves = [l for l in leaves if l[3]]
            nonterminal_leaves = [l for l in leaves if not l[3]]

            # 终局节点直接回传
            # _backup 从叶节点视角开始，自动交替翻转符号
            for node, state, path, is_term, term_val in terminal_leaves:
                self._backup(path, term_val, root)

            # 非终局节点 batch evaluate
            if nonterminal_leaves:
                states = [s for _, s, _, _, _ in nonterminal_leaves]
                policies, values = self.evaluator.evaluate_batch(states)

                for i, (node, state, path, _, _) in enumerate(nonterminal_leaves):
                    policy = policies[i]
                    value = float(values[i])

                    # 扩展节点
                    legal_moves = state.legal_moves()
                    legal_actions = ActionEncoder.legal_actions(legal_moves)
                    uniform_prior = 1.0 / max(len(legal_actions), 1)
                    for idx in legal_actions:
                        node.children[int(idx)] = MCTSNode(
                            prior=float(max(policy[idx], uniform_prior * 0.1)))

                    # _backup 从叶节点视角开始，自动交替翻转符号
                    self._backup(path, value, root)

            sims_done += batch_count

        # 从访问次数计算概率分布
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        total_visits = sum(c.visit_count for c in root.children.values())

        if total_visits == 0:
            for idx in root.children:
                policy[idx] = 1.0 / len(root.children)
            return policy

        if temperature == 0.0:
            best_idx = max(root.children.keys(),
                           key=lambda k: root.children[k].visit_count)
            policy[best_idx] = 1.0
        else:
            for idx, child in root.children.items():
                policy[idx] = child.visit_count ** (1.0 / temperature)
            s = policy.sum()
            if s > 0:
                policy /= s
            else:
                for idx in root.children:
                    policy[idx] = 1.0 / len(root.children)

        return policy

    def select_move(self, root_state: GameState,
                    temperature: float = 1.0) -> tuple:
        """搜索并采样一步走法。返回 (Move, policy)"""
        probs = self.search(root_state, temperature)
        legal_moves = root_state.legal_moves()
        if not legal_moves:
            return None, probs

        legal_actions = ActionEncoder.legal_actions(legal_moves)

        if temperature == 0.0:
            idx = int(np.argmax(probs))
        else:
            probs_legal = probs[legal_actions].copy()
            s = probs_legal.sum()
            if s > 0:
                probs_legal /= s
            else:
                probs_legal = np.ones(len(legal_actions)) / len(legal_actions)
            idx = int(legal_actions[np.random.choice(len(legal_actions), p=probs_legal)])

        move = ActionEncoder.decode(idx)
        return move, probs

    # ── 选择 ──

    def _select_leaf(self, root: MCTSNode, root_state: GameState):
        """选择一个叶节点。施加虚拟损失防止同batch重复选择。"""
        node = root
        state = root_state.copy()
        path = []

        while node.is_expanded() and not state.is_terminal():
            idx = node.select_child(self.c_puct)
            if idx < 0 or idx not in node.children:
                break
            path.append((node, idx))
            node = node.children[idx]
            # 虚拟损失：压低Q和inflate N，阻止同batch内重复选此路径
            node.visit_count += self.virtual_loss
            node.total_value -= self.virtual_loss
            move = ActionEncoder.decode(idx)
            if move is None:
                break
            state = state.apply(move)

        if state.is_terminal():
            r = state.game_result()
            if r is None:
                value = 0.0
            else:
                value = r.to_value(state.turn)
            path.append((node, -1))          # 叶节点加入path
            return (node, state, path, True, value)

        path.append((node, -1))              # 叶节点加入path
        return (node, state, path, False, 0.0)

    def _backup(self, path, value, root):
        """沿路径回传值。撤销虚拟损失，计入真实统计。

        path = [(root, a0), (child1, a1), ..., (leaf, -1)]
        value 为叶节点视角估值。从叶节点开始，每向上一层翻转符号。
        除 root 外所有节点均施加过虚拟损失，需要撤销。
        """
        n = len(path)
        vl = self.virtual_loss
        for i, (node, _) in enumerate(reversed(path)):
            at_root = (i == n - 1)  # root 是 reversed 的最后一个元素

            if at_root:
                # root 没有虚拟损失，直接计入真实统计
                node.visit_count += 1
                node.total_value += value
            else:
                # 撤销虚拟损失 + 计入真实值
                node.visit_count += 1 - vl
                node.total_value += value + vl

            value = -value           # 翻转为上一层视角

    # ── 扩展 ──

    def _expand(self, node: MCTSNode, state: GameState) -> float:
        """扩展节点的所有合法子节点。"""
        legal_moves = state.legal_moves()
        if not legal_moves:
            return 0.0

        policy, value = self.evaluator.evaluate(state)
        legal_actions = ActionEncoder.legal_actions(legal_moves)
        uniform_prior = 1.0 / len(legal_actions)

        for idx in legal_actions:
            node.children[int(idx)] = MCTSNode(
                prior=float(max(policy[idx], uniform_prior * 0.1)))
        return value
