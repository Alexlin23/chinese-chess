"""Leaf-batched AlphaZero PUCT with exactly-once reservation release."""
import numpy as np
from ..engine.constants import POLICY_SIZE
from ..engine.move import ActionEncoder
from .node import MCTSNode

class MCTS:
    def __init__(self, evaluator, num_simulations=800, c_puct=1.25,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 batch_size=1, virtual_loss=0.0, pb_c_base=19652.0, rng=None, reuse_tree=False,
                 cache_states=True, search_noise_depth=1):
        if num_simulations < 1 or batch_size < 1 or virtual_loss < 0:
            raise ValueError("invalid search budget/reservation")
        if type(search_noise_depth) is not int or search_noise_depth < 0:
            raise ValueError("search_noise_depth must be a nonnegative integer")
        self.evaluator, self.num_simulations = evaluator, num_simulations
        self.c_puct, self.pb_c_base = c_puct, pb_c_base
        self.dirichlet_alpha, self.dirichlet_epsilon = dirichlet_alpha, dirichlet_epsilon
        self.batch_size, self.virtual_loss = batch_size, virtual_loss
        self.rng = rng if rng is not None else np.random
        self.reuse_tree = reuse_tree
        self.cache_states = cache_states
        self.search_noise_depth = search_noise_depth
        self._noise_enabled = False
        self._prepared_nodes = set()
        self.last_root = None
        self._reuse_root = self._reuse_state = None
        self.diagnostics = {}

    def _release(self, path):
        for node, _ in path[1:]:
            node.visit_count -= self.virtual_loss
            node.total_value -= self.virtual_loss

    def _prepare_priors(self, node, depth):
        """Mix noise once per node/search; depth zero is the current root."""
        if not node.children or id(node) in self._prepared_nodes:
            return
        self._prepared_nodes.add(id(node))
        # A retained subtree may contain noise from a previous search. Always
        # start with its network priors, including when noise is now disabled.
        for child in node.children.values():
            child.prior = child.network_prior
        if self._noise_enabled and depth < self.search_noise_depth:
            noise = self.rng.dirichlet(np.full(len(node.children), self.dirichlet_alpha))
            for child, sample in zip(node.children.values(), noise):
                child.prior = ((1 - self.dirichlet_epsilon) * child.network_prior
                               + self.dirichlet_epsilon * float(sample))

    def _select_leaf(self, root, root_state, pending_nodes=None):
        node, state, path = root, root_state, []
        try:
            while node.is_expanded() and not state.is_terminal():
                self._prepare_priors(node, len(path))
                action = node.select_child(self.c_puct, self.pb_c_base)
                path.append((node, action))
                node = node.children[action]
                node.visit_count += self.virtual_loss
                node.total_value += self.virtual_loss
                if self.cache_states and node.rule_state is not None:
                    state = node.rule_state
                else:
                    state = state.apply_action(action)
                    if self.cache_states:
                        node.rule_state = state
            path.append((node, -1))
            if pending_nodes is not None and id(node) in pending_nodes:
                self._release(path)  # Includes the current leaf.
                return None
            result = state.game_result()
            return node, state, path, result is not None, (
                result.to_value(state.current_player) if result is not None else 0.0)
        except BaseException:
            if not path or path[-1][0] is not node:
                path.append((node, -1))
            self._release(path)
            raise

    def _backup(self, path, value, root):
        self._release(path)
        for node, _ in reversed(path):
            node.visit_count += 1
            node.total_value += value
            value = -value

    @staticmethod
    def _attach(node, state, policy):
        policy = np.asarray(policy, dtype=np.float64)
        if policy.shape != (POLICY_SIZE,):
            raise ValueError(f"neural policy must have shape ({POLICY_SIZE},)")
        actions = state.legal_actions()
        probabilities = policy[actions]
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise FloatingPointError("invalid neural priors")
        total = probabilities.sum()
        # Uniform fallback supports non-neural baseline evaluators with zero output.
        probabilities = probabilities / total if total > 0 else np.full(len(actions), 1 / len(actions))
        node.children = {int(a): MCTSNode(float(p)) for a, p in zip(actions, probabilities)}
        # Expanded nodes do not need their NN input again. Keep rule/legality
        # caches, but avoid retaining 47KB of planes for every visited state.
        state._encoding = None

    def _expand(self, node, state):
        policy, value = self.evaluator.evaluate(state)
        if not np.isfinite(value) or abs(value) > 1.00001:
            raise FloatingPointError("invalid neural value")
        self._attach(node, state, policy)
        return float(value)

    def search(self, root_state, temperature=1.0, add_exploration_noise=True):
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        output = np.zeros(POLICY_SIZE, dtype=np.float32)
        self.last_root = None
        self._noise_enabled = add_exploration_noise
        self._prepared_nodes = set()
        if root_state.is_terminal():
            return output
        reusable = (self.reuse_tree and self._reuse_root is not None and self._reuse_state is not None
                    and self._reuse_state.position_history == root_state.position_history
                    and self._reuse_state._max_ply == root_state._max_ply
                    and self._reuse_state.history == root_state.history
                    and self._reuse_state.repetition_counts == root_state.repetition_counts
                    and self._reuse_state.check_history == root_state.check_history
                    and self._reuse_state.no_capture_checks == root_state.no_capture_checks
                    and self._reuse_state.no_capture_count == root_state.no_capture_count)
        root = self._reuse_root if reusable else MCTSNode(1.0)
        self._reuse_root = self._reuse_state = None
        self.last_root = root
        # Remove the incoming leaf evaluation when turning a child into a root.
        root.visit_count = sum(c.visit_count for c in root.children.values())
        root.total_value = sum(-c.total_value for c in root.children.values())
        initial_visits = root.visit_count
        if not root.is_expanded():
            self._expand(root, root_state)
        self._prepare_priors(root, 0)
        completed, evaluated, collisions = 0, 0, 0
        while completed < self.num_simulations:
            outstanding = []
            pending = set()
            try:
                for _ in range(min(self.batch_size, self.num_simulations - completed)):
                    leaf = self._select_leaf(root, root_state, pending)
                    if leaf is None:
                        collisions += 1
                        break
                    outstanding.append(leaf)
                    if not leaf[3]:
                        pending.add(id(leaf[0]))
                if not outstanding:
                    raise RuntimeError("search made no progress")
                nonterminal = [leaf for leaf in outstanding if not leaf[3]]
                if nonterminal:
                    policies, values = self.evaluator.evaluate_batch([leaf[1] for leaf in nonterminal])
                    if (len(policies) != len(nonterminal) or len(values) != len(nonterminal)
                            or not np.isfinite(values).all() or np.any(np.abs(values) > 1.00001)):
                        raise FloatingPointError("invalid batch neural values")
                    neural = {id(leaf[0]): (p, float(v))
                              for leaf, p, v in zip(nonterminal, policies, values)}
                    evaluated += len(nonterminal)
                for leaf in list(outstanding):
                    node, state, path, terminal, value = leaf
                    if not terminal:
                        policy, value = neural[id(node)]
                        self._attach(node, state, policy)
                        self._prepare_priors(node, len(path) - 1)
                    self._backup(path, value, root)
                    outstanding.pop(0)
                    completed += 1
            finally:
                for leaf in outstanding:
                    self._release(leaf[2])
        count = sum(child.visit_count for child in root.children.values())
        if root.visit_count != initial_visits + self.num_simulations or count != root.visit_count:
            raise AssertionError("MCTS visit conservation violated")
        actions = np.array(list(root.children), dtype=np.int64)
        visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float64)
        if temperature == 0:
            output[max(root.children, key=lambda a: (root.children[a].visit_count, a))] = 1
        else:
            positive = visits > 0
            log_weights = np.log(visits[positive]) / temperature
            weights = np.exp(log_weights - log_weights.max())
            output[actions[positive]] = weights / weights.sum()
        self.diagnostics = dict(simulations=completed, evaluated_leaves=evaluated,
                                pending_collisions=collisions, root_visits=root.visit_count,
                                reused_visits=initial_visits)
        return output

    def select_move(self, root_state, temperature=1.0, add_exploration_noise=True):
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        # Training target always uses raw normalized root visits.
        pi = self.search(root_state, temperature=1.0,
                         add_exploration_noise=add_exploration_noise)
        if not pi.any():
            return None, pi
        if temperature == 0:
            action = max(self.last_root.children, key=lambda a: (self.last_root.children[a].visit_count, a))
        else:
            actions = np.flatnonzero(pi)
            logs = np.log(pi[actions].astype(np.float64)) / temperature
            probabilities = np.exp(logs - logs.max())
            probabilities /= probabilities.sum()
            action = int(self.rng.choice(actions, p=probabilities))
        move = ActionEncoder.decode(action)
        if self.reuse_tree:
            self._reuse_root = self.last_root.children[action]
            self._reuse_state = root_state.apply(move)
        return move, pi
