"""MCTS 树节点"""
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    """蒙特卡洛树搜索节点。

    每个节点代表"在当前局面下执行了某一步走法后到达的状态"。
    根节点是空节点（prior=1, visit_count=0），其 children 是第一步的所有合法走法。
    """
    prior: float                           # 先验概率 P(s, a)
    visit_count: int = 0                   # 访问次数 N(s, a)
    total_value: float = 0.0               # 累计价值 W(s, a)
    children: dict[int, 'MCTSNode'] = field(default_factory=dict)
    rule_state: object = field(default=None, repr=False, compare=False)

    @property
    def q(self) -> float:
        """平均价值 Q = W / N。未访问时返回 0。"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def select_child(self, c_puct: float, pb_c_base: float = 19652.0) -> int:
        """选择 PUCT 值最大的子节点。返回 move_index。"""
        best_idx = -1
        best_score = -float('inf')
        from math import log
        sqrt_parent = self.visit_count ** 0.5
        coefficient = c_puct + log((self.visit_count + pb_c_base + 1) / pb_c_base)

        for idx, child in self.children.items():
            # Q is stored from the child (next player's) perspective.
            # The parent must negate it before comparing its own actions.
            score = -child.q + coefficient * child.prior * sqrt_parent / (1.0 + child.visit_count)
            if score > best_score or (score == best_score and idx > best_idx):
                best_score = score
                best_idx = idx

        return best_idx

    def is_expanded(self) -> bool:
        """是否已扩展（拥有子节点）。"""
        return len(self.children) > 0
