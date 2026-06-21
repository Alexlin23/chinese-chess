"""贪心 + α-β — 冷启动数据生成器

使用快速启发式评估驱动对弈，生成大量带真实终局信号的训练数据。
策略层: 1-ply 贪心 (试所有走法, 选评分最高的)
速度层: 直接在 fast_chess 上操作, 避免 GameState 封装开销

速度: ~2s/局 (1-ply), ~70%+ 杀棋率
用途: 训练神经网络学会基本价值预测后, 切换回 MCTS 自对弈
"""
import numpy as np
import time
from typing import Optional

from ..engine.state import GameState
from ..engine.move import Move, ActionEncoder
from ..engine.fast_chess import make_move, undo_move, is_in_check, get_valid_moves
from ..engine.constants import ROWS, COLS
from ..search.evaluator import HeuristicEvaluator


class GreedyPlayer:
    """1层贪心引擎 — 直接操作 numpy board 避免拷贝开销。

    对每个合法走法做 原地模拟→评估→撤销, 用 softmax 采样选择走法。
    temperature 控制随机性: >0 时更好走法概率更高, →0 时贪心。
    """

    def __init__(self, seed: int = 42, temperature: float = 0.5):
        self.evaluator = HeuristicEvaluator(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature

    def select_move(self, state: GameState) -> Optional[Move]:
        """softmax 采样选择。temperature=0 退化为纯贪心。"""
        legal = state.legal_moves()
        if not legal:
            return None

        # 纯贪心
        if self.temperature <= 0:
            return self._greedy_pick(state, legal)

        # 计算每个走法的评分
        scores = np.zeros(len(legal), dtype=np.float32)
        turn = state.turn
        board = state.board
        evaluator = self.evaluator

        for i, move in enumerate(legal):
            undo_info = make_move(board,
                                  (move.from_row, move.from_col),
                                  (move.to_row, move.to_col))
            captured = undo_info["captured"]
            nmc = state.move_count + 1
            nnc = 0 if captured else state.no_capture_count + 1

            tmp = GameState(board, not turn, nmc, nnc)
            value = evaluator.evaluate(tmp)[1]
            scores[i] = -value

            undo_move(board, undo_info)

        # Softmax 概率
        scores_centered = scores - scores.max()
        scaled = scores_centered / max(self.temperature, 0.01)
        exp_scores = np.exp(np.clip(scaled, -50, 50))
        probs = exp_scores / exp_scores.sum()

        idx = self.rng.choice(len(legal), p=probs)
        return legal[idx]

    def _greedy_pick(self, state, legal):
        """纯贪心选择。"""
        best_move = legal[0]
        best_score = -float('inf')
        turn = state.turn
        board = state.board
        evaluator = self.evaluator

        for move in legal:
            undo_info = make_move(board,
                                  (move.from_row, move.from_col),
                                  (move.to_row, move.to_col))
            captured = undo_info["captured"]
            nmc = state.move_count + 1
            nnc = 0 if captured else state.no_capture_count + 1

            tmp = GameState(board, not turn, nmc, nnc)
            value = evaluator.evaluate(tmp)[1]
            score = -value

            undo_move(board, undo_info)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move


class AlphaBetaPlayer:
    """轻量 α-β 搜索 — depth=2 时探索己方走法 + 对方应着。

    比纯贪心多算一层对弈, 但用快速原地模拟避免 GameState 分配。
    每步: 己方~40步 × 对方~40步 × evaluate = ~1600 次评估。
    每步 ≈ 0.3s, 一局 ≈ 30s。
    """

    def __init__(self, depth: int = 2, seed: int = 42):
        self.depth = depth
        self.evaluator = HeuristicEvaluator(seed=seed)
        self.rng = np.random.default_rng(seed)

    def select_move(self, state: GameState) -> Optional[Move]:
        """α-β depth=2 搜索选择走法。"""
        legal = state.legal_moves()
        if not legal:
            return None

        best_move = legal[0]
        best_score = -float('inf')
        turn = state.turn
        board = state.board

        for move in legal:
            undo_main = make_move(board,
                                  (move.from_row, move.from_col),
                                  (move.to_row, move.to_col))

            # 对方最优应着
            captured = undo_main["captured"]
            next_mc = state.move_count + 1
            next_nc = 0 if captured else state.no_capture_count + 1

            if self.depth <= 1:
                tmp = GameState(board, not turn, next_mc, next_nc)
                opp_value = self.evaluator.evaluate(tmp)[1]
                score = -opp_value  # 对方视角→己方视角
            else:
                # depth=2: 找对方最优应着
                opp_moves = get_valid_moves(board, None, None, not turn)
                opp_moves_list = self._moves_from_dicts(opp_moves, board, not turn)
                opp_best = float('inf')  # 对方最小化

                for opp_move in opp_moves_list:
                    undo_opp = make_move(board,
                                         (opp_move.from_row, opp_move.from_col),
                                         (opp_move.to_row, opp_move.to_col))

                    oc = undo_opp["captured"]
                    omc = next_mc + 1
                    onc = 0 if oc else next_nc + 1
                    tmp = GameState(board, turn, omc, onc)
                    val = self.evaluator.evaluate(tmp)[1]
                    # val 从己方视角

                    undo_move(board, undo_opp)
                    if val < opp_best:
                        opp_best = val
                    if len(opp_moves_list) > 30 and val < -0.5:
                        break  # 早停: 对方已有很好的应着

                score = opp_best

            undo_move(board, undo_main)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _moves_from_dicts(self, moves, board, turn):
        """从 fast_chess dict 格式转为 Move 列表 (只取合法走法)。"""
        result = []
        rows, cols = np.where(board * (1 if turn else -1) > 0)
        seen = set()
        for r, c in zip(rows, cols):
            valid = get_valid_moves(board, r, c, turn)
            for m in valid:
                key = (r, c, m["row"], m["col"])
                if key not in seen:
                    seen.add(key)
                    result.append(Move(*key))
        return result


# ── 自对弈 ──

class GreedySelfPlay:
    """贪心自对弈数据生成器。

    产出与 SelfPlayGame.get_training_data() 格式一致的训练样本。
    Args:
        temperature: softmax 温度 (>0 = 有随机性, 0 = 纯贪心)
    """

    def __init__(self, max_moves: int = 120, seed: int = 42,
                 temperature: float = 0.5, use_depth2: bool = False):
        self.max_moves = max_moves
        if use_depth2:
            self.player = AlphaBetaPlayer(depth=2, seed=seed)
        else:
            self.player = GreedyPlayer(seed=seed, temperature=temperature)
        self.rng = self.player.rng

    def play_one(self) -> dict:
        """进行一局自对弈。"""
        t0 = time.perf_counter()
        state = GameState.new_game()
        positions: list[tuple[np.ndarray, bool]] = []

        for _ in range(self.max_moves):
            if state.is_terminal():
                break
            move = self.player.select_move(state)
            if move is None:
                break
            positions.append((state.encode(), state.turn))
            state = state.apply(move)

        final_result = state.result() or 0.0
        elapsed = time.perf_counter() - t0

        data = []
        for state_enc, is_red in positions:
            value = final_result if is_red else -final_result
            policy = np.zeros(ActionEncoder.POLICY_SIZE, dtype=np.float32)
            data.append((state_enc.copy(), policy.copy(), value))

        winner = ('RED' if final_result > 0.5 else
                  ('BLACK' if final_result < -0.5 else 'DRAW'))

        return {
            'steps': state.move_count,
            'winner': winner,
            'result': final_result,
            'time': elapsed,
            'data': data,
        }


def generate_bootstrap_data(num_games: int = 100,
                            max_moves: int = 120,
                            seed: int = 42,
                            temperature: float = 0.5,
                            use_depth2: bool = False,
                            verbose: bool = True) -> tuple:
    """批量生成冷启动训练数据。

    Args:
        num_games:    对局数
        max_moves:    单局最大步数
        seed:         随机种子
        temperature:  softmax 温度 (越高越随机, 0=纯贪心)
        use_depth2:   是否用 depth=2 α-β (更慢但更准)
        verbose:      是否打印进度

    Returns:
        (all_data, stats): all_data 是 [(state, policy, value), ...]
    """
    engine = GreedySelfPlay(max_moves=max_moves, seed=seed,
                           temperature=temperature, use_depth2=use_depth2)
    all_data = []
    stats = {'red': 0, 'black': 0, 'draw': 0, 'total_steps': 0}

    t0 = time.perf_counter()
    for i in range(num_games):
        info = engine.play_one()

        all_data.extend(info['data'])
        if info['result'] > 0.5:
            stats['red'] += 1
        elif info['result'] < -0.5:
            stats['black'] += 1
        else:
            stats['draw'] += 1
        stats['total_steps'] += info['steps']

        if verbose:
            w = info['winner']
            print(f"  G{i+1:3d}: {info['steps']:3d} steps, {w:5s}  "
                  f"[{info['time']*1000:.0f}ms]")

    elapsed = time.perf_counter() - t0
    terminal = stats['red'] + stats['black']

    if verbose:
        print(f"\nDone: {elapsed:.1f}s ({elapsed/num_games:.0f}ms/game)")
        print(f"  RED={stats['red']}, BLACK={stats['black']}, DRAW={stats['draw']}")
        print(f"  Terminal: {terminal}/{num_games} = "
              f"{terminal/num_games*100:.0f}%")
        print(f"  Samples: {len(all_data)}")

    return all_data, stats


# ── 不对称对弈（强 vs 弱）— 大量终局信号 ──

class RandomPlayer:
    """纯随机走子 — 弱方，用于不平衡对弈。"""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def select_move(self, state: GameState) -> Optional[Move]:
        legal = state.legal_moves()
        return self.rng.choice(legal) if legal else None


def generate_asymmetric_games(num_games: int = 200,
                              max_moves: int = 100,
                              seed: int = 42,
                              strong_is_red: bool = None,
                              verbose: bool = True) -> tuple:
    """不平衡对弈：贪心 vs 随机，产出 > 90% 杀棋率。

    强方用贪心 1-ply 搜索，弱方纯随机。
    强方各执红/黑各半（rand_strong_is_red 随机分配消除偏置）。

    Returns:
        (all_data, stats)
    """
    strong_rng = np.random.default_rng(seed)
    greedy_red = GreedyPlayer(seed=seed, temperature=0.2)
    greedy_black = GreedyPlayer(seed=seed + 1, temperature=0.2)
    random_red = RandomPlayer(seed=seed)
    random_black = RandomPlayer(seed=seed + 1)

    all_data = []
    stats = {'red_wins': 0, 'black_wins': 0, 'draws': 0}

    t0 = time.perf_counter()
    for i in range(num_games):
        if strong_is_red is None:
            strong_red = strong_rng.random() > 0.5
        else:
            strong_red = strong_is_red
        seed_offset = i * 100

        state = GameState.new_game()
        positions: list[tuple[np.ndarray, bool]] = []

        for step in range(max_moves):
            if state.is_terminal():
                break

            is_red_turn = state.turn
            if is_red_turn:
                player = greedy_red if strong_red else random_red
            else:
                player = greedy_black if not strong_red else random_black

            move = player.select_move(state)
            if move is None:
                break

            positions.append((state.encode(), is_red_turn))
            state = state.apply(move)

        final_result = state.result() or 0.0

        for state_enc, is_red in positions:
            value = final_result if is_red else -final_result
            policy = np.zeros(ActionEncoder.POLICY_SIZE, dtype=np.float32)
            all_data.append((state_enc.copy(), policy.copy(), value))

        if final_result > 0.5:
            stats['red_wins'] += 1
        elif final_result < -0.5:
            stats['black_wins'] += 1
        else:
            stats['draws'] += 1

        if verbose and (i + 1) % 20 == 0:
            t = time.perf_counter() - t0
            term = stats['red_wins'] + stats['black_wins']
            print(f"  G{i+1:3d}/{num_games}: {term}/{i+1} term "
                  f"(R:{stats['red_wins']} B:{stats['black_wins']}) [{t:.0f}s]")

    elapsed = time.perf_counter() - t0
    terminal = stats['red_wins'] + stats['black_wins']

    if verbose:
        print(f"\nAsymmetric: {elapsed:.0f}s ({elapsed/num_games:.0f}ms/game)")
        print(f"  RED wins:   {stats['red_wins']}")
        print(f"  BLACK wins: {stats['black_wins']}")
        print(f"  Draws:      {stats['draws']}")
        print(f"  Terminal:   {terminal}/{num_games} = "
              f"{terminal/num_games*100:.0f}%")
        print(f"  Samples:    {len(all_data)}")
        # 非零 value 比例
        nz = sum(1 for _, _, v in all_data if abs(v) > 0.01)
        print(f"  Non-zero values: {nz}/{len(all_data)} = "
              f"{nz/len(all_data)*100:.0f}%")

    return all_data, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="贪心 α-β 冷启动数据生成")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--depth2", action="store_true",
                       help="使用 depth=2 α-β")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asymmetric", action="store_true",
                       help="不对称对弈 (强 vs 弱)")
    args = parser.parse_args()

    if args.asymmetric:
        data, stats = generate_asymmetric_games(
            num_games=args.games, seed=args.seed)
    else:
        data, stats = generate_bootstrap_data(
            num_games=args.games, seed=args.seed, use_depth2=args.depth2)
