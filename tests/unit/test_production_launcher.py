from __future__ import annotations

import signal
import threading
from unittest.mock import MagicMock

from src.production import launcher


def test_child_commands_do_not_reenter_launcher_in_source_mode(monkeypatch):
    monkeypatch.delattr(launcher.sys, "frozen", raising=False)
    assert launcher._child_command("api")[-2:] == ["--child", "api"]
    assert launcher._child_command("worker")[-2:] == ["--child", "worker"]
    assert launcher._child_command("dashboard")[-2:] == ["--child", "dashboard"]


def test_seed_runtime_assets_preserve_existing_admin_edits(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    packaged = bundle / "config"
    packaged.mkdir(parents=True)
    (packaged / "settings.yaml").write_text("packaged", encoding="utf-8")
    runtime = tmp_path / "runtime"
    (runtime / "config").mkdir(parents=True)
    target = runtime / "config" / "settings.yaml"
    target.write_text("admin-edit", encoding="utf-8")
    monkeypatch.setattr(launcher.sys, "_MEIPASS", str(bundle), raising=False)
    launcher._seed_runtime_assets(runtime)
    assert target.read_text(encoding="utf-8") == "admin-edit"


def test_seed_runtime_assets_copy_public_review_candidates(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    packaged = bundle / "evaluation" / "candidates"
    corpus = packaged / "public_corpus"
    corpus.mkdir(parents=True)
    (packaged / "public_golden_candidate.json").write_text("{}", encoding="utf-8")
    (corpus / "evidence.md").write_text("evidence", encoding="utf-8")
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(launcher.sys, "_MEIPASS", str(bundle), raising=False)

    launcher._seed_runtime_assets(runtime)

    assert (
        runtime / "evaluation" / "candidates" / "public_golden_candidate.json"
    ).read_text(encoding="utf-8") == "{}"
    assert (
        runtime / "evaluation" / "candidates" / "public_corpus" / "evidence.md"
    ).read_text(encoding="utf-8") == "evidence"


def test_worker_child_clears_launcher_arguments(monkeypatch):
    observed = []
    monkeypatch.setattr("src.production.worker.main", lambda: observed.append(list(launcher.sys.argv)))
    launcher.sys.argv = ["launcher.exe", "--child", "worker"]
    launcher._run_child("worker")
    assert observed == [[launcher.sys.executable]]


def test_launcher_waits_for_liveness_not_production_readiness(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "_app_root", lambda: tmp_path)
    monkeypatch.setattr(launcher, "_port_available", lambda _port: True)
    monkeypatch.setattr(launcher, "_seed_runtime_assets", lambda _root: None)
    monkeypatch.setattr("src.core.settings.load_settings", lambda: MagicMock(
        server=MagicMock(port=8766, dashboard_port=8501),
    ))
    processes = [MagicMock(), MagicMock(), MagicMock()]
    processes[0].poll.side_effect = [None, 0, 0, 0]
    for process in processes[1:]:
        process.poll.return_value = None
    monkeypatch.setattr(launcher.subprocess, "Popen", MagicMock(side_effect=processes))
    response = MagicMock(status_code=200)
    get = MagicMock(return_value=response)
    monkeypatch.setattr(launcher.httpx, "get", get)
    monkeypatch.setattr(launcher.webbrowser, "open", lambda _url: True)
    monkeypatch.setattr(launcher, "_create_process_job", lambda _processes: None)

    launcher.main()

    get.assert_called_once_with("http://127.0.0.1:8766/health/live", timeout=1)


def test_launcher_can_disable_browser_for_service_or_automated_runs(monkeypatch):
    monkeypatch.setenv("RAG_OPEN_BROWSER", "0")
    assert launcher._should_open_browser() is False


def test_launcher_opens_browser_by_default(monkeypatch):
    monkeypatch.delenv("RAG_OPEN_BROWSER", raising=False)
    assert launcher._should_open_browser() is True


def test_launcher_exposes_native_qdrant_manager_in_packaged_exe(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(launcher, "_app_root", lambda: tmp_path)
    monkeypatch.setattr(launcher, "_seed_runtime_assets", lambda _root: None)
    monkeypatch.setattr("src.production.native_qdrant.main", lambda args: observed.append(args) or 0)
    monkeypatch.setattr(
        launcher.sys,
        "argv",
        [
            "ModularRAGLauncher.exe",
            "--qdrant",
            "status",
            "--qdrant-base-dir",
            str(tmp_path / "qdrant"),
        ],
    )

    launcher.main()

    assert observed == [
        ["status", "--timeout", "90.0", "--base-dir", str(tmp_path / "qdrant")]
    ]


def test_stop_signal_sets_event_without_raising(monkeypatch):
    installed = {}
    monkeypatch.setattr(launcher.signal, "getsignal", lambda signum: f"old-{signum}")
    monkeypatch.setattr(
        launcher.signal, "signal", lambda signum, handler: installed.setdefault(signum, handler),
    )
    stop_event = threading.Event()

    previous = launcher._install_stop_handlers(stop_event)
    installed[signal.SIGINT](signal.SIGINT, None)

    assert stop_event.is_set()
    assert previous[signal.SIGINT] == f"old-{signal.SIGINT}"


def test_stop_processes_uses_one_shared_deadline_and_closes_job(monkeypatch):
    processes = [MagicMock(), MagicMock(), MagicMock()]
    for process in processes:
        process.poll.return_value = None
    processes[-1].wait.side_effect = launcher.subprocess.TimeoutExpired("child", 1)
    job = MagicMock()
    now = iter((100.0, 101.0, 102.0, 103.0))
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(now))
    monkeypatch.setattr(launcher.sys, "platform", "win32")

    launcher._stop_processes(processes, job, timeout=10)

    for process in processes:
        process.send_signal.assert_called_once_with(signal.CTRL_BREAK_EVENT)
    processes[-1].kill.assert_called_once()
    assert processes[-1].wait.call_args.kwargs["timeout"] <= 10
    assert processes[-2].wait.call_args.kwargs["timeout"] < processes[-1].wait.call_args.kwargs["timeout"]
    job.close.assert_called_once_with()
