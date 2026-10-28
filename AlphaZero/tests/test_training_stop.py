"""Stopping must save a consistent learner and retire all owned workers."""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

import psutil
import pytest
import torch
import yaml

from AlphaZero.train.config import AlphaZeroConfig
from AlphaZero.train.live import LiveTraining


def post(url, body, origin=None, action="stop"):
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
    request = urllib.request.Request(url + "/api/" + action, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.load(response)


def test_stop_requires_current_token_and_same_origin_and_is_idempotent():
    live = LiveTraining(port=0)
    calls = []
    url = "http://%s:%s" % live.address
    try:
        token = live.control_token
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(url, {"token": token})
        assert exc.value.code == 409  # Not enabled until startup finishes.
        live.enable_stop(lambda: calls.append("stop"))
        for body, origin in (({}, None), ({"token": "old-session"}, None),
                             ({"token": token}, "https://other.example")):
            with pytest.raises(urllib.error.HTTPError) as exc:
                post(url, body, origin)
            assert exc.value.code == 403
        assert not calls
        assert post(url, {"token": token}, url)[0] == 202
        assert post(url, {"token": token})[0] == 202
        assert calls == ["stop"]
        assert live.snapshot()["phase"] == "stopping"
    finally:
        live.close()


def tiny_config(**changes):
    return AlphaZeroConfig(num_blocks=1, num_filters=4, batch_size=2,
        device="cpu", precision="fp32", num_workers=1, actor_processes=1,
        num_simulations=1, max_game_ply=4, checkpoint_interval=20, recovery_interval=10,
        max_optimizer_steps=20, max_iterations=1, telemetry_port=0, **changes)


def assert_saved(directory, step):
    learner = torch.load(directory / "learner.pt", map_location="cpu", weights_only=False)
    actor = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
    stop = json.loads((directory / "training-stop.json").read_text())
    assert stop["saved"] and stop["step"] == learner["step"] == actor["step"] == step
    assert stop["run_id"] == learner["run_id"] == actor["run_id"]
    assert all(torch.equal(v, actor["model_state_dict"][k]) and bool(torch.isfinite(v).all())
               for k, v in learner["model_state_dict"].items())
    assert learner["extra"]["replay_generation"] == stop["completed_games"]


def test_http_stop_while_waiting_releases_pipeline_and_children(tmp_path):
    config = tiny_config(new_positions_per_update=10000000)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    log = tmp_path / "stdout.log"
    with log.open("w") as output:
        process = subprocess.Popen([sys.executable, "-u", "-m", "AlphaZero.scripts.pipeline",
            "--config", str(config_path), "--output-dir", str(tmp_path)], stdout=output, stderr=output)
    children = []
    try:
        deadline = time.monotonic() + 35
        url = None
        while time.monotonic() < deadline:
            assert process.poll() is None, log.read_text()
            for line in log.read_text().splitlines():
                if line.startswith("Training dashboard: "):
                    url = line.removeprefix("Training dashboard: ")
            if url:
                with urllib.request.urlopen(url + "/api/snapshot", timeout=3) as response:
                    state = json.load(response)
                if state.get("stop_available") and state.get("boards"):
                    break
            time.sleep(.1)
        else:
            pytest.fail("startup timed out: " + log.read_text())
        children = psutil.Process(process.pid).children(recursive=True)
        assert len(children) >= 2
        assert post(url, {"token": state["control_token"]})[0] == 202
        assert process.wait(timeout=15) == 0, log.read_text()
        # multiprocessing's tracker can retire just after its parent's exit.
        deadline = time.monotonic() + 3
        while any(child.is_running() for child in children) and time.monotonic() < deadline:
            time.sleep(.05)
        assert not [child for child in children if child.is_running()]
        assert_saved(tmp_path, 0)
        assert "RuntimeError" not in log.read_text()
    finally:
        if process.poll() is None:
            for child in psutil.Process(process.pid).children(recursive=True):
                child.kill()
            process.kill()
            process.wait(timeout=5)


def test_stop_during_sgd_finishes_and_saves_unpublished_update(tmp_path, monkeypatch):
    from AlphaZero.scripts.pipeline import pipeline
    from AlphaZero.train.trainer import Trainer
    enable = LiveTraining.enable_stop
    step = Trainer.step
    callback = []
    def remember(self, handler):
        callback.append(handler)
        enable(self, handler)
    def stop_after_update(self, replay):
        result = step(self, replay)
        callback[0]()  # Stop requested while the outer SGD call is still in progress.
        return result
    monkeypatch.setattr(LiveTraining, "enable_stop", remember)
    monkeypatch.setattr(Trainer, "step", stop_after_update)
    result = pipeline(tiny_config(), output_dir=tmp_path)
    assert result["step"] == 1  # Not the regular publication boundary (20).
    assert_saved(tmp_path, 1)


def test_dashboard_stop_resume_cycles_preserve_checkpoint_and_gate(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    from AlphaZero.scripts.pipeline import pipeline
    from AlphaZero.scripts.training_dashboard import TrainingDashboard
    # Make a real small CPU checkpoint, independent of the user's run.
    seed = replace(tiny_config(), checkpoint_interval=1, max_optimizer_steps=1)
    pipeline(seed, output_dir=tmp_path, enable_monitor=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(tiny_config(new_positions_per_update=10000000).to_dict()))
    dashboard = TrainingDashboard(str(config_path), str(tmp_path))
    url = "http://%s:%s" % dashboard.live.address
    token = dashboard.live.control_token
    def until(predicate):
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            dashboard.tick()
            state = dashboard.live.snapshot()
            if predicate(state):
                return state
            time.sleep(.05)
        pytest.fail("dashboard transition timed out: " + str(dashboard.live.snapshot()))
    try:
        assert dashboard.process is None
        assert dashboard.live.snapshot()["resume_available"]
        gate = None
        for _ in range(2):
            # Concurrent clicks are idempotent: only one child pipeline can be started.
            with ThreadPoolExecutor(max_workers=4) as pool:
                statuses = list(pool.map(lambda _: post(url, {"token": token}, action="resume")[0], range(4)))
            assert statuses == [202] * 4
            state = until(lambda s: s.get("stop_available") and bool(s.get("boards")))
            root = psutil.Process(dashboard.process.pid)
            owned = [root] + root.children(recursive=True)
            assert state["step"] == 1 and state["actor_step"] == 1
            assert post(url, {"token": token})[0] == 202
            assert post(url, {"token": token})[0] == 202
            paused = until(lambda s: s.get("resume_available"))
            assert paused["phase"] == "stopped" and paused["saved"]
            assert not paused.get("terminal") and not paused["boards"]
            assert not [p for p in owned if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
            assert_saved(tmp_path, 1)
            learner = torch.load(tmp_path / "learner.pt", map_location="cpu", weights_only=False)
            actual_gate = learner["extra"]["update_schedule"]
            assert gate is None or actual_gate == gate
            gate = actual_gate
            with urllib.request.urlopen(url + "/api/snapshot", timeout=3) as response:
                assert json.load(response)["resume_available"]  # The same page still responds.
        assert dashboard.process is None  # No automatic restart after stop.
    finally:
        dashboard.close()


def test_dashboard_process_does_not_import_torch():
    subprocess.run([sys.executable, "-c",
        "import sys; import AlphaZero.scripts.training_dashboard; assert 'torch' not in sys.modules"],
        check=True, timeout=10)
