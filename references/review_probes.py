"""Targeted review evidence; run from the repository root in the review WSL env."""
import ast
import gc
import importlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from AlphaZero.engine import GameState
from AlphaZero.engine.constants import KING, KNIGHT, POLICY_SIZE, ROOK
from AlphaZero.engine.move import ActionEncoder, Move
from AlphaZero.engine.repetition import position_key
from AlphaZero.search import MCTS
from AlphaZero.search.node import MCTSNode
from AlphaZero.train.benchmark import evaluate_agents, random_agent
from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.replay import ReplayBuffer, pack_sample


def numpy_probes(out):
    b = np.zeros((10, 9), dtype=np.int8)
    for r, c, p in [(0, 4, -KING), (9, 3, KING), (1, 0, ROOK),
                    (2, 3, ROOK), (2, 6, KNIGHT)]:
        b[r, c] = p
    move = Move(1, 0, 1, 4)
    a = GameState(b, True, max_ply=1)
    child = a.apply(move)
    normal = GameState(b.copy(), True, max_ply=512).apply(move)
    out['mate_on_limit'] = dict(legal=a.is_legal(move),
        opponent_legal_moves=len(child.legal_moves()),
        capped_result=child.game_result().__dict__,
        normal_result=normal.game_result().__dict__)

    state = GameState.new_game()
    key = position_key(state.board, state.turn)
    a = GameState(state.board, True, 8, 8, {key: 1})
    b = GameState(state.board.copy(), True, 8, 8, {key: 2})
    equal = np.array_equal(a.encode(), b.encode())
    for m in [Move(9, 1, 7, 2), Move(0, 1, 2, 2),
              Move(7, 2, 9, 1), Move(2, 2, 0, 1)]:
        a, b = a.apply(m), b.apply(m)
    out['history_alias'] = dict(same_initial_encoding=equal,
        result_one_visit=None if a.game_result() is None else a.game_result().__dict__,
        result_two_visits=None if b.game_result() is None else b.game_result().__dict__)

    report = evaluate_agents(random_agent, random_agent, 2, max_ply=4, seed=7)
    out['all_draw_ci'] = dict(draws=report.draws, interval=report.score_ci95)

    class Peaked:
        def evaluate(self, s):
            p = np.zeros(POLICY_SIZE, np.float32)
            p[s.legal_actions()[0]] = 1
            return p, 0.
    root = MCTSNode(1.)
    MCTS(Peaked())._expand(root, state)
    out['prior_mass'] = sum(c.prior for c in root.children.values())

    # Execute the actual board conversion function without starting the web app.
    module = ast.parse((ROOT / 'backend/main.py').read_text(encoding='utf-8'))
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_board_to_gamestate')
    scope = {'_AlphaGameState': GameState}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'backend/main.py', 'exec'), scope)
    names = {1: '帥', 2: '仕', 3: '相', 4: '馬', 5: '車', 6: '炮', 7: '兵'}
    played = GameState.new_game()
    for m in [Move(9, 1, 7, 2), Move(0, 1, 2, 2),
              Move(7, 2, 9, 1), Move(2, 2, 0, 1)]:
        played = played.apply(m)
    board = [[None if p == 0 else {'type': names[abs(int(p))],
              'color': 'r' if p > 0 else 'b'} for p in row] for row in played.board]
    converted = scope['_board_to_gamestate'](board, 'r')
    out['web_state_reset'] = dict(expected_move_count=played.move_count,
        expected_no_capture_count=played.no_capture_count,
        expected_repetition_positions=len(played.repetition_counts),
        move_count=converted.move_count,
        no_capture_count=converted.no_capture_count,
        repetition_positions=len(converted.repetition_counts))


def torch_probes(out):
    import torch
    from AlphaZero.model import PolicyWDLEncoder
    from AlphaZero.train.trainer import Trainer
    from AlphaZero.train.inference_server import InferenceServer, BatchRequest
    torch.set_num_threads(1)
    torch.manual_seed(7)
    out['torch_version'] = torch.__version__
    out['cuda_available'] = torch.cuda.is_available()
    pipe = importlib.import_module('AlphaZero.scripts.pipeline')
    cfg = AlphaZeroConfig(num_blocks=1, num_filters=4, num_simulations=2,
        mcts_batch_size=1, inference_batch_size=2, num_workers=2,
        max_game_ply=4, batch_size=2, samples_per_epoch=4,
        epochs_per_iteration=1, grad_accum_steps=1)

    class StopProbe(Exception):
        pass

    captured = {}
    class CapturedTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured['trainer'] = self

    def inspect_selfplay(**kwargs):
        checkpoint = torch.load(kwargs['model_path'], weights_only=False)
        t = captured['trainer']
        out[captured['case']] = dict(current_iteration=t.current_iteration,
            tensors_different=sum(not torch.equal(v, t.model.state_dict()[k])
                for k, v in checkpoint['model_state_dict'].items()),
            old_files=sorted(p.name for p in Path(kwargs['model_path']).parent.glob('*_old.pt')))
        raise StopProbe()

    with tempfile.TemporaryDirectory() as directory:
        with patch.object(pipe, 'Trainer', CapturedTrainer), patch.object(pipe, 'parallel_self_play', inspect_selfplay):
            captured['case'] = 'fresh_initialization'
            try:
                pipe.pipeline(cfg, 1, output_dir=directory, enable_monitor=False)
            except StopProbe:
                pass
            iter_path = Path(directory) / 'iter049.pt'
            torch.save({'model_state_dict': captured['trainer'].model.state_dict(),
                        'config': cfg.to_dict()}, iter_path)
            (Path(directory) / 'iter000_old.pt').unlink()
            captured['case'] = 'resume_iteration_file'
            try:
                pipe.pipeline(cfg, 1, resume=str(iter_path), output_dir=directory, enable_monitor=False)
            except StopProbe:
                pass
    captured.clear()
    gc.collect()

    t = Trainer(cfg, 'cpu')
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.nn.Parameter(torch.zeros(1))
            self.register_buffer('direction', torch.nn.functional.one_hot(torch.tensor(0), POLICY_SIZE).float())
        def forward(self, x):
            return (self.x * self.direction).expand(len(x), -1), (self.x * 0).expand(len(x), 3)
    class FixedReplay:
        def __len__(self): return 1
        def sample(self, n):
            p = np.zeros((n, POLICY_SIZE), np.float32)
            p[:, 0] = 1
            w = np.zeros((n, 3), np.float32)
            w[:, 1] = 1
            return np.zeros((n, 18, 10, 9), np.float32), p, w, np.ones(n, np.float32)
    t.model = Toy()
    t.optimizer = torch.optim.SGD(t.model.parameters(), lr=0.)
    t.scheduler = torch.optim.lr_scheduler.StepLR(t.optimizer, 100000)
    t.config.samples_per_epoch, t.config.batch_size, t.config.grad_accum_steps = 3, 1, 2
    gradients = []
    handle = t.optimizer.register_step_pre_hook(lambda *args: gradients.append(float(t.model.x.grad)))
    t.train(FixedReplay(), epochs=1)
    handle.remove()
    out['gradient_remainder'] = dict(gradients=gradients, ratio=gradients[-1] / gradients[0],
        optimizer_steps=len(gradients), scheduler_steps=t.scheduler.last_epoch)
    del t

    cfg.batch_size, cfg.samples_per_epoch, cfg.grad_accum_steps = 2, 4, 1
    with tempfile.TemporaryDirectory() as directory:
        trainer = Trainer(cfg, 'cpu')
        old_adam = torch.optim.Adam(trainer.model.parameters())
        old_path = str(Path(directory) / 'adam.pt')
        torch.save({'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': old_adam.state_dict()}, old_path)
        try:
            trainer.load_checkpoint(old_path)
            for p in trainer.model.parameters():
                p.grad = torch.zeros_like(p)
            trainer.optimizer.step()
            out['adam_migration'] = 'step succeeded'
        except Exception as exc:
            out['adam_migration'] = repr(exc)
        del trainer, old_adam
        gc.collect()

        model = PolicyWDLEncoder(num_blocks=1, num_filters=4)
        checkpoint_path = str(Path(directory) / 'model.pt')
        torch.save({'model_state_dict': model.state_dict(), 'config': cfg.to_dict()}, checkpoint_path)
        arena_module = importlib.import_module('AlphaZero.train.arena')
        traces = []
        class TracedSearch(MCTS):
            def select_move(self, *args, **kwargs):
                move, policy = super().select_move(*args, **kwargs)
                traces[-1].append([move.from_row, move.from_col, move.to_row, move.to_col])
                return move, policy
        class TracedArena(arena_module.Arena):
            def _play_game(self, *args):
                traces.append([])
                return super()._play_game(*args)
        with patch.object(arena_module, 'MCTS', TracedSearch):
            TracedArena(cfg, 'cpu').evaluate(checkpoint_path, checkpoint_path, num_games=4, verbose=False)
        out['arena_repetition'] = dict(games=len(traces),
            same_red_trajectory=traces[0] == traces[2],
            same_black_trajectory=traces[1] == traces[3],
            unique_trajectories=len({json.dumps(t) for t in traces}))
        server = InferenceServer(checkpoint_path, 'cpu', batch_size=2, max_wait_ms=20, num_workers=2)
        server.start()
        try:
            state = GameState.new_game()
            for worker in [0, 1]:
                server.request_queue.put(BatchRequest(worker + 1, state.encode()[None], state.legal_mask()[None], worker))
            responses = [server.response_queues[w].get(timeout=45) for w in [0, 1]]
            expected = model.evaluate_batch(torch.from_numpy(state.encode()[None]), torch.from_numpy(state.legal_mask()[None]))[0]
            out['cpu_inference'] = dict(errors=[r.error for r in responses],
                ids=[r.request_id for r in responses],
                max_policy_error=max(float(np.max(np.abs(r.policies - expected))) for r in responses))
        finally:
            server.stop()
        replay = pipe.parallel_self_play(cfg, checkpoint_path, 2, 1, device='cpu', verbose=False)
        out['cpu_selfplay'] = dict(samples=len(replay))
        trainer = Trainer(cfg, 'cpu')
        stats = trainer.train(replay, epochs=1)
        out['cpu_training'] = dict(loss=float(stats['avg_loss']), batches=stats['batches'])
        restore_path = str(Path(directory) / 'restore.pt')
        trainer.save_checkpoint(restore_path)
        other = Trainer(cfg, 'cpu')
        other.load_checkpoint(restore_path)
        out['cpu_restore'] = dict(iteration=other.current_iteration,
            equal_parameters=all(torch.equal(v, other.model.state_dict()[k]) for k,v in trainer.model.state_dict().items()))


def comprehensive_probes(out):
    """Additional cross-module checks for the comprehensive review."""
    import copy
    import hashlib
    import time
    from types import SimpleNamespace
    import torch
    from AlphaZero.model import PolicyWDLEncoder
    from AlphaZero.train.trainer import Trainer
    from AlphaZero.train.arena_stats import promotion_evidence
    from AlphaZero.engine.state import GameResult
    from AlphaZero.engine.constants import BLACK
    torch.set_num_threads(1)
    torch.manual_seed(11)
    np.random.seed(11)
    pipe = importlib.import_module('AlphaZero.scripts.pipeline')
    cfg = AlphaZeroConfig(num_blocks=1, num_filters=4, num_simulations=2,
        mcts_batch_size=1, inference_batch_size=2, num_workers=1,
        max_game_ply=4, batch_size=2, samples_per_epoch=4,
        epochs_per_iteration=1, grad_accum_steps=1)

    def digest(state):
        return hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in state.values())).hexdigest()
    generators, candidates = [], []
    def fake_selfplay(**kwargs):
        ck = torch.load(kwargs['model_path'], weights_only=False)
        generators.append(digest(ck['model_state_dict']))
        replay = kwargs['replay']
        state = GameState.new_game()
        p = state.legal_mask().astype(np.float32)
        p /= p.sum()
        for _ in range(2): replay.add(state, p, np.array([0, 1, 0], np.float32))
        return replay
    def all_draw_evaluation(self, candidate, best):
        candidates.append(digest(torch.load(candidate, weights_only=False)['model_state_dict']))
        score, lower, promote = promotion_evidence(0, 0, 400, 0.55)
        return dict(wins=0, losses=0, draws=400, score_rate=score,
                    decisive_win_rate_lower_95=lower, should_promote=promote)
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(pipe, 'parallel_self_play', fake_selfplay), patch.object(pipe.Arena, 'evaluate', all_draw_evaluation):
            pipe.pipeline(cfg, 2, output_dir=directory, enable_monitor=False)
        final = digest(torch.load(Path(directory) / 'latest.pt', weights_only=False)['model_state_dict'])
        out['rollback_learning_gate'] = dict(rounds=2,
            same_generator_in_both_rounds=generators[0] == generators[1],
            candidate_changes_discarded=all(c != generators[0] for c in candidates),
            final_equals_initial_generator=final == generators[0],
            note='Real SGD training; controlled all-draw Arena results, not a strength result')
    gc.collect()

    counts = {}
    for label, kwargs in [('default', dict(num_blocks=4, num_filters=64)),
                          ('test_yaml', dict(num_blocks=20, num_filters=256))]:
        model = PolicyWDLEncoder(**kwargs)
        sizes = {'body': 0, 'policy': 0, 'value': 0}
        for name, p in model.named_parameters():
            key = 'policy' if name.startswith('policy_') else 'value' if name.startswith('wdl_') else 'body'
            sizes[key] += p.numel()
        sizes['total'] = sum(sizes.values())
        counts[label] = sizes
        del model
    out['parameter_counts'] = counts
    cfg_history = AlphaZeroConfig(input_channels=131, num_blocks=1, num_filters=4)
    trainer = Trainer(cfg_history, 'cpu')
    out['ignored_input_channels'] = dict(config=cfg_history.input_channels,
                                        actual=trainer.model.conv_input.in_channels)
    del trainer
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'schedule.yaml'
        path.write_text('training:\n  lr_decay_step: 3\n  lr_decay_rate: 0.5\nself_play:\n  repetition_limit: 5\n')
        c = AlphaZeroConfig.from_yaml(str(path))
        out['ignored_yaml_fields'] = dict(lr_decay_step=c.lr_decay_step,
            lr_decay_rate=c.lr_decay_rate, repetition_limit=c.repetition_limit)

    logits = torch.zeros(3, requires_grad=True)
    target = torch.tensor([1., 0., 0.])
    full_loss = -(target * logits.log_softmax(0)).sum()
    full_grad = torch.autograd.grad(full_loss, logits)[0]
    masked_loss = -(target[:2] * logits[:2].log_softmax(0)).sum()
    masked_grad = torch.autograd.grad(masked_loss, logits)[0]
    masked = logits.masked_fill(torch.tensor([False, False, True]), -float('inf'))
    naive_loss = -(target * masked.log_softmax(0)).sum()
    out['policy_probability_contract'] = dict(full_loss=float(full_loss.detach()),
        legal_only_loss=float(masked_loss.detach()), full_gradient=full_grad.tolist(),
        legal_only_gradient=masked_grad.tolist(), naive_masked_loss_is_nan=bool(torch.isnan(naive_loss)))

    class Uniform:
        def __init__(self): self.batch_sizes = []
        def evaluate(self, s):
            p = s.legal_mask().astype(np.float32)
            return p / p.sum(), 0.
        def evaluate_batch(self, states):
            self.batch_sizes.append(len(states))
            return np.stack([self.evaluate(s)[0] for s in states]), np.zeros(len(states))
    class TracedMCTS(MCTS):
        def _expand(self, node, state):
            self.observed_root = node
            return super()._expand(node, state)
    budgets = []
    for batch in [1, 16, 64]:
        evaluator = Uniform()
        search = TracedMCTS(evaluator, num_simulations=64, batch_size=batch)
        p = search.search(GameState.new_game(), add_exploration_noise=False)
        root = search.observed_root
        stack, visits = [root], []
        while stack:
            node = stack.pop()
            visits.append(node.visit_count)
            stack.extend(node.children.values())
        budgets.append(dict(batch=batch, root_visits=root.visit_count,
            child_visits=sum(c.visit_count for c in root.children.values()),
            root_visit_conservation=root.visit_count == sum(c.visit_count for c in root.children.values()),
            all_counts_nonnegative_integers=all(v >= 0 and float(v).is_integer() for v in visits),
            network_batches=evaluator.batch_sizes, policy_sum=float(p.sum())))
    out['mcts_budget_accounting'] = budgets
    forced_board = np.zeros((10, 9), np.int8)
    for r, c, p in [(0, 4, -KING), (9, 3, KING), (1, 0, ROOK), (2, 3, ROOK)]:
        forced_board[r, c] = p
    forced = GameState(forced_board, False)
    evaluator = Uniform()
    search = TracedMCTS(evaluator, num_simulations=64, batch_size=16)
    search.search(forced, add_exploration_noise=False)
    out['mcts_forced_reply_reservations'] = dict(legal_moves=len(forced.legal_moves()),
        root_visits=search.observed_root.visit_count,
        child_visits=sum(c.visit_count for c in search.observed_root.children.values()),
        network_batches=evaluator.batch_sizes)
    terminal = GameState.new_game(max_ply=0)
    move, p = MCTS(Uniform(), num_simulations=2).select_move(terminal, temperature=0.)
    out['terminal_select_move'] = dict(terminal=terminal.is_terminal(),
        move=str(move), legal=terminal.is_legal(move), policy_sum=float(p.sum()))
    out['color_contract'] = dict(black_constant=BLACK,
        constructor_black_becomes_red=GameState(GameState.new_game().board, BLACK).turn,
        black_win_for_false=GameResult(BLACK, 'fixture').to_value(False),
        black_win_for_black=GameResult(BLACK, 'fixture').to_value(BLACK))
    out['action_roundtrip'] = all(ActionEncoder.encode(ActionEncoder.decode(i)) == i
                                  for i in range(POLICY_SIZE))

    state = GameState.new_game()
    policy = state.legal_mask().astype(np.float32)
    policy /= policy.sum()
    invalid = policy.copy()
    invalid[state.legal_actions()[0]] = -0.2
    sample = pack_sample(state, invalid, np.array([0, 1, 0], np.float32))
    out['replay_negative_policy'] = dict(accepted=True, stored_negative=bool(np.any(sample.probabilities < 0)))
    sample = pack_sample(state, policy, np.array([0, 1, 0], np.float32))
    shallow = sum(sys.getsizeof(x) for x in [sample, sample.__dict__, sample.board,
                                            sample.actions, sample.probabilities])
    out['sample_byte_accounting'] = dict(reported=sample.nbytes,
        shallow_object_bytes_excluding_scalars=shallow)
    with tempfile.TemporaryDirectory() as directory:
        replay = ReplayBuffer()
        replay.add_packed(sample)
        path = Path(directory) / 'corrupt.npz'
        replay.save(str(path))
        with np.load(path) as data: arrays = {k: data[k] for k in data.files}
        arrays['probabilities'][:] = 0
        np.savez_compressed(path, **arrays)
        corrupt = ReplayBuffer.load(str(path))
        with np.errstate(invalid='ignore', divide='ignore'):
            restored = corrupt.sample(1)
        out['corrupt_replay'] = dict(load_succeeded=True,
            sampled_policy_finite=bool(np.isfinite(restored[1]).all()))
    timings = {}
    for n in [1000, 100000]:
        replay = ReplayBuffer(max_size=n)
        for _ in range(n): replay.add_packed(sample)
        elapsed = []
        for _ in range(5):
            start = time.perf_counter()
            replay.sample(64)
            elapsed.append((time.perf_counter() - start) * 1000)
        timings[str(n)] = float(np.median(elapsed))
    out['replay_sampling_median_ms'] = timings

    db = importlib.import_module('backend.database')
    rules = importlib.import_module('backend.chess_rules')
    module = ast.parse((ROOT / 'backend/main.py').read_text(encoding='utf-8'))
    function = copy.deepcopy(next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'api_move'))
    function.decorator_list = []
    scope = dict(MoveRequest=object, MoveResponse=lambda **kw: SimpleNamespace(**kw),
        PieceInfo=lambda **kw: SimpleNamespace(**kw), _parse_board=lambda b: b,
        _serialize_board=lambda b: b, get_step_count=db.get_step_count, record_step=db.record_step,
        is_valid_move=rules.is_valid_move, make_move=rules.make_move,
        check_game_result=rules.check_game_result, is_in_check=rules.is_in_check)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'backend/main.py', 'exec'), scope)
    with tempfile.TemporaryDirectory() as directory, patch.object(db, 'DB_PATH', str(Path(directory) / 'review.db')):
        db.init_db()
        initial = rules.create_initial_board()
        gid = db.create_game(initial)
        def request(column):
            return SimpleNamespace(board=copy.deepcopy(initial), turn='r', game_id=gid,
                from_pos=SimpleNamespace(row=6, col=column), to_pos=SimpleNamespace(row=5, col=column))
        first = scope['api_move'](request(0))
        stale = scope['api_move'](request(2))
        game = db.get_game(gid)
        out['stale_web_move'] = dict(first_accepted=first.valid, stale_accepted=stale.valid,
            colors=[m['piece_color'] for m in db.get_moves(gid)],
            first_move_undone_in_board=game['board'][6][0] is not None and game['board'][5][0] is None)
        before_board = copy.deepcopy(game['board'])
        with patch.object(db, 'update_game', side_effect=RuntimeError('simulated failure before board update')):
            try: db.undo_game(gid, rules.create_initial_board)
            except RuntimeError: pass
        out['undo_nonatomic'] = dict(remaining_moves=len(db.get_moves(gid)),
            board_unchanged=db.get_game(gid)['board'] == before_board)


def movement_and_resilience_probes(out):
    from AlphaZero.engine.fast_chess import get_raw_moves, is_valid_move
    from AlphaZero.engine.constants import CANNON, PAWN, ELEPHANT, ADVISOR
    from AlphaZero.tests.test_rule_differential import backend_moves
    import torch
    from AlphaZero.train.inference_server import InferenceServer, BatchRequest
    from AlphaZero.model import PolicyWDLEncoder
    checks = []
    fixtures = []
    def raw_fixture(name, pieces, origin, destination, expected):
        board = np.zeros((10,9), np.int8)
        for r,c,p in pieces: board[r,c] = p
        fixtures.append((name, board, origin, destination, expected))
    raw_fixture('knight_leg', [(5,4,KNIGHT),(6,4,PAWN)], (5,4), (7,5), False)
    raw_fixture('knight_free', [(5,4,KNIGHT)], (5,4), (7,5), True)
    raw_fixture('cannon_screen', [(5,0,CANNON),(5,2,PAWN),(5,4,-ROOK)], (5,0), (5,4), True)
    raw_fixture('cannon_no_screen', [(5,0,CANNON),(5,4,-ROOK)], (5,0), (5,4), False)
    raw_fixture('cannon_two_screens', [(5,0,CANNON),(5,2,PAWN),(5,3,PAWN),(5,4,-ROOK)], (5,0), (5,4), False)
    raw_fixture('elephant_river', [(5,2,ELEPHANT)], (5,2), (3,4), False)
    raw_fixture('elephant_eye', [(7,2,ELEPHANT),(6,3,PAWN)], (7,2), (5,4), False)
    raw_fixture('advisor_palace', [(7,3,ADVISOR)], (7,3), (6,2), False)
    raw_fixture('king_palace', [(7,4,KING)], (7,4), (6,4), False)
    raw_fixture('pawn_before_river', [(6,4,PAWN)], (6,4), (6,5), False)
    raw_fixture('pawn_after_river', [(4,4,PAWN)], (4,4), (4,5), True)
    raw_fixture('pawn_backwards', [(4,4,PAWN)], (4,4), (5,4), False)
    raw_fixture('rook_block', [(5,0,ROOK),(5,2,PAWN),(5,4,-PAWN)], (5,0), (5,4), False)
    for name, board, origin, dest, expected in fixtures:
        for rotated in [False, True]:
            b = -board[::-1,::-1].copy() if rotated else board.copy()
            fr,fc = (9-origin[0],8-origin[1]) if rotated else origin
            tr,tc = (9-dest[0],8-dest[1]) if rotated else dest
            actual = any(m['row']==tr and m['col']==tc for m in get_raw_moves(b,fr,fc))
            checks.append(dict(name=name, rotated=rotated, passed=bool(actual==expected)))
    face = np.zeros((10,9), np.int8)
    face[0,4],face[9,4],face[5,4] = -KING,KING,ROOK
    checks.append(dict(name='exposing_facing_kings', rotated=False,
        passed=not is_valid_move(face,(5,4),(5,3),True)))
    out['manual_movement_fixtures'] = dict(checks=len(checks),
        failures=[c for c in checks if not c['passed']])
    rng = np.random.default_rng(21015)
    state = GameState.new_game(max_ply=120)
    mismatches = 0
    for _ in range(300):
        if state.is_terminal(): state=GameState.new_game(max_ply=120)
        legal=state.legal_moves()
        current={(m.from_row,m.from_col,m.to_row,m.to_col) for m in legal}
        mismatches += current != backend_moves(state)
        state = state.apply(legal[int(rng.integers(len(legal)))])
    out['extended_rule_differential'] = dict(positions=300, mismatches=mismatches,
        scope='Two in-repository implementations; not an independent rules oracle')
    with tempfile.TemporaryDirectory() as directory:
        model = PolicyWDLEncoder(num_blocks=1,num_filters=4)
        with torch.no_grad(): next(model.parameters()).view(-1)[0] = float('nan')
        path = str(Path(directory)/'nonfinite.pt')
        torch.save(dict(model_state_dict=model.state_dict(),config=dict(num_blocks=1,num_filters=4)),path)
        server=InferenceServer(path,'cpu',batch_size=2,num_workers=1)
        server.start()
        try:
            s=GameState.new_game()
            server.request_queue.put(BatchRequest(1,s.encode()[None],s.legal_mask()[None],0))
            response=server.response_queues[0].get(timeout=45)
            out['nonfinite_inference_response'] = dict(error=response.error,
                finite_policies=bool(np.isfinite(response.policies).all()),
                finite_values=bool(np.isfinite(response.values).all()))
        finally: server.stop()


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    evidence = {}
    numpy_probes(evidence)
    try:
        torch_probes(evidence)
        comprehensive_probes(evidence)
        movement_and_resilience_probes(evidence)
    except ImportError as exc:
        evidence['torch_unavailable'] = repr(exc)
    path = ROOT / 'references/review_evidence.json'
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
