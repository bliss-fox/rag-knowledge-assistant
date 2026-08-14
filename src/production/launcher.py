"""Windows-friendly launcher for API, worker and Streamlit dashboard."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Sequence

import httpx


class _WindowsKillOnCloseJob:
    """Own a Windows Job Object that terminates children when the launcher exits."""

    def __init__(self, processes: Sequence[subprocess.Popen[Any]]) -> None:
        import ctypes
        from ctypes import wintypes

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        )
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._kernel32 = kernel32
        self._handle = handle
        try:
            limits = _ExtendedLimitInformation()
            limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                handle, 9, ctypes.byref(limits), ctypes.sizeof(limits),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            for process in processes:
                process_handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
                if not kernel32.AssignProcessToJobObject(handle, process_handle):
                    raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _create_process_job(
    processes: Sequence[subprocess.Popen[Any]],
) -> _WindowsKillOnCloseJob | None:
    if sys.platform != "win32":
        return None
    try:
        return _WindowsKillOnCloseJob(processes)
    except OSError as exc:
        # Graceful shutdown remains available even if a restricted host prevents
        # nested Job Objects. Surface the missing crash-safety guarantee in logs.
        print(f"Warning: child process crash guard unavailable: {exc}", file=sys.stderr)
        return None


def _install_stop_handlers(stop_event: threading.Event) -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        except (OSError, ValueError):
            continue
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (OSError, ValueError):
            pass


def _stop_processes(
    processes: Sequence[subprocess.Popen[Any]],
    job: _WindowsKillOnCloseJob | None,
    timeout: float = 10.0,
) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.send_signal(
                    signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM
                )
            except (OSError, ProcessLookupError):
                pass
    deadline = time.monotonic() + timeout
    for process in reversed(processes):
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
    if job is not None:
        job.close()

def _port_available(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _child_command(mode: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--child", mode]
    if mode in {"api", "worker", "dashboard"}:
        return [sys.executable, "-m", "src.production.launcher", "--child", mode]
    raise ValueError(f"Unknown child mode: {mode}")


def _seed_runtime_config(root: Path) -> None:
    """Copy packaged defaults once; never overwrite administrator edits."""
    packaged = Path(getattr(sys, "_MEIPASS", root)) / "config"
    if not packaged.is_dir():
        return
    target = root / "config"
    for source in packaged.rglob("*"):
        if not source.is_file():
            continue
        destination = target / source.relative_to(packaged)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _run_child(mode: str) -> None:
    if mode == "api":
        from src.production.api import main as api_main
        sys.argv = [sys.executable]
        api_main()
        return
    if mode == "worker":
        from src.production.worker import main as worker_main
        sys.argv = [sys.executable]
        worker_main()
        return
    if mode == "dashboard":
        from streamlit.web.cli import main as streamlit_main

        from src.core.settings import load_settings

        settings = load_settings()
        dashboard_path = Path(getattr(sys, "_MEIPASS", _app_root())) / "src/observability/dashboard/app.py"
        sys.argv = [
            "streamlit", "run", str(dashboard_path),
            "--server.address", settings.server.host,
            "--server.port", str(settings.server.dashboard_port),
            "--server.headless", "true",
            "--global.developmentMode", "false",
            "--browser.gatherUsageStats", "false",
        ]
        streamlit_main()
        return
    raise SystemExit(f"Unknown child mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--child", choices=("api", "worker", "dashboard"))
    args, _ = parser.parse_known_args()
    root = _app_root()
    os.environ.setdefault("RAG_APP_ROOT", str(root))
    _seed_runtime_config(root)
    if args.child:
        _run_child(args.child)
        return

    from src.core.settings import load_settings

    settings = load_settings()
    api_port = settings.server.port
    dashboard_port = settings.server.dashboard_port
    for port in (api_port, dashboard_port):
        if not _port_available(port):
            raise SystemExit(f"Port {port} is already in use")
    flags = (
        subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        if sys.platform == "win32" else 0
    )
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    api_log = (logs / "api.log").open("a", encoding="utf-8")
    worker_log = (logs / "worker.log").open("a", encoding="utf-8")
    dashboard_log = (logs / "dashboard.log").open("a", encoding="utf-8")
    processes = [
        subprocess.Popen(_child_command("api"), cwd=root, creationflags=flags,
                         stdout=api_log, stderr=subprocess.STDOUT),
        subprocess.Popen(_child_command("worker"), cwd=root, creationflags=flags,
                         stdout=worker_log, stderr=subprocess.STDOUT),
        subprocess.Popen(_child_command("dashboard"), cwd=root, creationflags=flags,
                         stdout=dashboard_log, stderr=subprocess.STDOUT),
    ]
    process_job = _create_process_job(processes)
    stop_event = threading.Event()
    previous_handlers = _install_stop_handlers(stop_event)
    try:
        startup_timeout = float(os.environ.get("RAG_STARTUP_TIMEOUT_SECONDS", "60"))
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("A service exited before readiness")
            try:
                if httpx.get(f"http://127.0.0.1:{api_port}/health/live", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"API liveness timeout after {startup_timeout:g}s; see logs/api.log"
            )
        webbrowser.open(f"http://127.0.0.1:{dashboard_port}")
        print(f"RAG services are running. Dashboard: http://127.0.0.1:{dashboard_port}")
        while not stop_event.wait(1) and all(process.poll() is None for process in processes):
            pass
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        _restore_signal_handlers(previous_handlers)
        _stop_processes(processes, process_job)
        for handle in (api_log, worker_log, dashboard_log):
            handle.close()


if __name__ == "__main__":
    main()
