"""Install and manage the pinned native Qdrant release on Windows.

The manager intentionally owns only the project-specific Qdrant process and
never removes its storage directory.  A PID is actionable only after the
running process image has been resolved and matched to the immutable install
receipt, preventing a stale PID file from terminating an unrelated process.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

QDRANT_VERSION = "1.15.4"
QDRANT_ARCHIVE_URL = (
    "https://github.com/qdrant/qdrant/releases/download/"
    "v1.15.4/qdrant-x86_64-pc-windows-msvc.zip"
)
QDRANT_ARCHIVE_SHA256 = "5348af719d5d9f3658597c8d3a8231bb8c38e7139a8076b69bd3dd59151d06d5"
QDRANT_HTTP_URL = "http://127.0.0.1:6333"
_RECEIPT_NAME = "install-receipt.json"
_PID_NAME = "qdrant.pid.json"
_MAX_LOG_BYTES = 20 * 1024 * 1024


class NativeQdrantError(RuntimeError):
    """Expected lifecycle failure with a stable machine-readable category."""

    def __init__(self, message: str, *, category: str) -> None:
        super().__init__(message)
        self.category = category


def default_base_dir() -> Path:
    """Return the per-user application directory used by the native service."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise NativeQdrantError(
            "LOCALAPPDATA is unavailable; pass --base-dir explicitly.",
            category="local_app_data_unavailable",
        )
    return Path(local_app_data) / "ModularRAG" / "qdrant"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_extract_archive(archive: Path, destination: Path) -> None:
    """Extract a trusted-hash ZIP while still rejecting unsafe member paths."""
    destination.mkdir(parents=True, exist_ok=False)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            normalized = member.filename.replace("\\", "/")
            relative = PurePosixPath(normalized)
            if (
                not normalized
                or relative.is_absolute()
                or ".." in relative.parts
                or (relative.parts and ":" in relative.parts[0])
            ):
                raise NativeQdrantError(
                    f"Qdrant archive contains an unsafe path: {member.filename!r}",
                    category="unsafe_archive",
                )
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise NativeQdrantError(
                    f"Qdrant archive contains a symbolic link: {member.filename!r}",
                    category="unsafe_archive",
                )
            target = destination.joinpath(*relative.parts)
            try:
                target.resolve().relative_to(resolved_destination)
            except ValueError as exc:
                raise NativeQdrantError(
                    f"Qdrant archive escapes the staging directory: {member.filename!r}",
                    category="unsafe_archive",
                ) from exc
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _download_archive(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ModularRAG-Qdrant-Manager/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    except (OSError, urllib.error.URLError) as exc:
        raise NativeQdrantError(
            "Unable to download the pinned Qdrant archive.",
            category="download_failed",
        ) from exc


def _health_probe() -> bool:
    try:
        with urllib.request.urlopen(f"{QDRANT_HTTP_URL}/healthz", timeout=1.0) as response:
            return int(response.status) == 200
    except (OSError, urllib.error.URLError):
        return False


def _port_in_use() -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", 6333)) == 0


def _query_process_image_windows(pid: int) -> Path | None:
    """Resolve a process image without relying on deprecated WMIC output."""
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == error_invalid_parameter:
            return None
        raise ctypes.WinError(error)
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buffer))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        return Path(buffer.value)
    finally:
        kernel32.CloseHandle(handle)


def _terminate_process_windows(pid: int) -> None:
    os.kill(pid, signal.SIGTERM)


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(str(first.resolve(strict=False))) == os.path.normcase(
        str(second.resolve(strict=False))
    )


class NativeQdrantManager:
    """Lifecycle manager for one pinned, loopback-only native Qdrant process."""

    def __init__(
        self,
        *,
        base_dir: Path | None = None,
        storage_dir: Path | None = None,
        downloader: Callable[[str, Path], None] = _download_archive,
        health_probe: Callable[[], bool] = _health_probe,
        port_in_use: Callable[[], bool] = _port_in_use,
        process_image: Callable[[int], Path | None] = _query_process_image_windows,
        terminate_process: Callable[[int], None] = _terminate_process_windows,
        popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        platform_name: str = sys.platform,
    ) -> None:
        self.base_dir = (base_dir or default_base_dir()).resolve(strict=False)
        self.install_dir = self.base_dir / f"v{QDRANT_VERSION}"
        self.storage_dir = (storage_dir or self.base_dir / "storage").resolve(strict=False)
        self.runtime_dir = self.base_dir / "runtime"
        self.logs_dir = self.base_dir / "logs"
        self.receipt_path = self.install_dir / _RECEIPT_NAME
        self.pid_path = self.runtime_dir / _PID_NAME
        self.log_path = self.logs_dir / "qdrant.log"
        self.downloader = downloader
        self.health_probe = health_probe
        self.port_in_use = port_in_use
        self.process_image = process_image
        self.terminate_process = terminate_process
        self.popen_factory = popen_factory
        self.sleep = sleep
        self.monotonic = monotonic
        self.platform_name = platform_name

    def _require_windows(self) -> None:
        if self.platform_name != "win32":
            raise NativeQdrantError(
                "The native Qdrant manager supports Windows only; use Compose on other hosts.",
                category="unsupported_platform",
            )

    def install(self, *, force: bool = False) -> dict[str, Any]:
        self._require_windows()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        backup: Path | None = None
        if self.install_dir.exists() and not force:
            try:
                executable = self._validated_executable()
            except NativeQdrantError as exc:
                raise NativeQdrantError(
                    "The existing Qdrant installation is invalid; rerun install with --force.",
                    category="invalid_existing_install",
                ) from exc
            return self._install_result("already_installed", executable)

        current = self.status()
        if current["state"] in {"running", "unhealthy", "pid_conflict", "verification_failed"}:
            raise NativeQdrantError(
                "Stop the managed Qdrant process before replacing its executable.",
                category="process_running",
            )

        with tempfile.TemporaryDirectory(prefix=".qdrant-install-", dir=self.base_dir) as temp:
            temporary = Path(temp)
            archive = temporary / "qdrant.zip"
            extracted = temporary / "payload"
            self.downloader(QDRANT_ARCHIVE_URL, archive)
            actual_archive_sha = _sha256_file(archive)
            if actual_archive_sha != QDRANT_ARCHIVE_SHA256:
                raise NativeQdrantError(
                    "The downloaded Qdrant archive failed SHA-256 verification.",
                    category="archive_hash_mismatch",
                )
            _safe_extract_archive(archive, extracted)
            executables = [path for path in extracted.rglob("qdrant.exe") if path.is_file()]
            if len(executables) != 1:
                raise NativeQdrantError(
                    "The pinned archive must contain exactly one qdrant.exe.",
                    category="invalid_archive_layout",
                )
            executable = executables[0]
            executable_relative = executable.relative_to(extracted).as_posix()
            receipt = {
                "schema_version": 1,
                "product": "qdrant",
                "version": QDRANT_VERSION,
                "archive_url": QDRANT_ARCHIVE_URL,
                "archive_sha256": QDRANT_ARCHIVE_SHA256,
                "executable": executable_relative,
                "executable_sha256": _sha256_file(executable),
                "installed_at": _utc_now(),
            }
            _atomic_write_json(extracted / _RECEIPT_NAME, receipt)

            if self.install_dir.exists():
                backup = self.base_dir / (
                    f"v{QDRANT_VERSION}.backup-{datetime.now().strftime('%Y%m%d%H%M%S')}-"
                    f"{uuid.uuid4().hex[:8]}"
                )
                os.replace(self.install_dir, backup)
            try:
                os.replace(extracted, self.install_dir)
            except BaseException:
                if backup is not None and backup.exists() and not self.install_dir.exists():
                    os.replace(backup, self.install_dir)
                raise

        installed = self._validated_executable()
        result = self._install_result("installed", installed)
        if backup is not None:
            result["previous_install_backup"] = str(backup)
        return result

    def start(self, *, timeout: float = 90.0) -> dict[str, Any]:
        self._require_windows()
        if timeout <= 0:
            raise NativeQdrantError(
                "Startup timeout must be positive.", category="invalid_timeout"
            )
        executable = self._validated_executable()
        status = self.status()
        if status["state"] == "running" and status["healthy"]:
            status["action"] = "already_running"
            return status
        if status["state"] not in {"stopped"}:
            raise NativeQdrantError(
                f"Cannot start Qdrant while lifecycle state is {status['state']!r}.",
                category=str(status.get("error_category") or "unsafe_process_state"),
            )
        if self.port_in_use():
            raise NativeQdrantError(
                "Port 6333 is occupied by a process that is not this managed Qdrant instance.",
                category="port_conflict",
            )

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._rotate_log_if_needed()
        environment = os.environ.copy()
        environment.update(
            {
                "QDRANT__SERVICE__HOST": "127.0.0.1",
                "QDRANT__SERVICE__HTTP_PORT": "6333",
                "QDRANT__SERVICE__GRPC_PORT": "6334",
                "QDRANT__STORAGE__STORAGE_PATH": str(self.storage_dir),
                "QDRANT__TELEMETRY_DISABLED": "true",
            }
        )
        flags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
        with self.log_path.open("ab", buffering=0) as log_handle:
            process = self.popen_factory(
                [str(executable)],
                cwd=str(self.install_dir),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        _atomic_write_json(
            self.pid_path,
            {
                "schema_version": 1,
                "pid": int(process.pid),
                "executable": str(executable),
                "started_at": _utc_now(),
            },
        )
        deadline = self.monotonic() + timeout
        while self.monotonic() < deadline:
            if process.poll() is not None:
                self.pid_path.unlink(missing_ok=True)
                raise NativeQdrantError(
                    "Qdrant exited before becoming healthy; inspect qdrant.log.",
                    category="process_exited",
                )
            if self.health_probe():
                return {
                    "ok": True,
                    "action": "started",
                    "state": "running",
                    "healthy": True,
                    "pid": int(process.pid),
                    "version": QDRANT_VERSION,
                    "endpoint": QDRANT_HTTP_URL,
                    "storage_preserved": True,
                }
            self.sleep(0.25)
        try:
            process.terminate()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        finally:
            self.pid_path.unlink(missing_ok=True)
        raise NativeQdrantError(
            f"Qdrant did not become healthy within {timeout:g} seconds; inspect qdrant.log.",
            category="startup_timeout",
        )

    def stop(self, *, timeout: float = 15.0) -> dict[str, Any]:
        self._require_windows()
        if timeout <= 0:
            raise NativeQdrantError("Stop timeout must be positive.", category="invalid_timeout")
        record = self._load_pid_record()
        if record is None:
            if self.health_probe():
                raise NativeQdrantError(
                    "A Qdrant-compatible service is running without this manager's PID record.",
                    category="unmanaged_process",
                )
            return {
                "ok": True,
                "action": "already_stopped",
                "state": "stopped",
                "healthy": False,
                "storage_preserved": True,
            }
        pid, _actual = self._verified_process(record)
        if pid is None:
            self.pid_path.unlink(missing_ok=True)
            return {
                "ok": True,
                "action": "stale_pid_cleared",
                "state": "stopped",
                "healthy": False,
                "storage_preserved": True,
            }
        self.terminate_process(pid)
        deadline = self.monotonic() + timeout
        while self.monotonic() < deadline:
            try:
                actual = self.process_image(pid)
            except OSError:
                # Windows may deny a new query handle while a verified process is
                # exiting.  Once both health and the listening port are gone, the
                # manager has enough evidence to clear only its own stale record.
                if not self.health_probe() and not self.port_in_use():
                    actual = None
                else:
                    self.sleep(0.1)
                    continue
            if actual is None or not _same_path(actual, _actual):
                self.pid_path.unlink(missing_ok=True)
                return {
                    "ok": True,
                    "action": "stopped",
                    "state": "stopped",
                    "healthy": False,
                    "pid": pid,
                    "storage_preserved": True,
                }
            self.sleep(0.1)
        raise NativeQdrantError(
            "The verified Qdrant process did not stop within the configured timeout.",
            category="stop_timeout",
        )

    def restart(self, *, timeout: float = 90.0) -> dict[str, Any]:
        self.stop(timeout=min(timeout, 15.0))
        result = self.start(timeout=timeout)
        result["action"] = "restarted"
        return result

    def status(self) -> dict[str, Any]:
        installed = True
        try:
            self._validated_executable()
        except NativeQdrantError as exc:
            installed = False
            install_error = exc.category
        else:
            install_error = None
        record = self._load_pid_record()
        if record is None:
            healthy = self.health_probe()
            return {
                "ok": installed and not healthy,
                "state": "unmanaged_running" if healthy else "stopped",
                "healthy": healthy,
                "installed": installed,
                "version": QDRANT_VERSION,
                "endpoint": QDRANT_HTTP_URL,
                "error_category": "unmanaged_process" if healthy else install_error,
                "storage_preserved": True,
            }
        try:
            pid, _actual = self._verified_process(record)
        except NativeQdrantError as exc:
            return {
                "ok": False,
                "state": "pid_conflict" if exc.category == "pid_path_mismatch" else "verification_failed",
                "healthy": self.health_probe(),
                "installed": installed,
                "version": QDRANT_VERSION,
                "endpoint": QDRANT_HTTP_URL,
                "error_category": exc.category,
                "storage_preserved": True,
            }
        if pid is None:
            self.pid_path.unlink(missing_ok=True)
            healthy = self.health_probe()
            return {
                "ok": installed and not healthy,
                "state": "unmanaged_running" if healthy else "stopped",
                "healthy": healthy,
                "installed": installed,
                "version": QDRANT_VERSION,
                "endpoint": QDRANT_HTTP_URL,
                "error_category": "unmanaged_process" if healthy else install_error,
                "storage_preserved": True,
            }
        healthy = self.health_probe()
        return {
            "ok": installed and healthy,
            "state": "running" if healthy else "unhealthy",
            "healthy": healthy,
            "installed": installed,
            "pid": pid,
            "version": QDRANT_VERSION,
            "endpoint": QDRANT_HTTP_URL,
            "error_category": None if healthy else "health_check_failed",
            "storage_preserved": True,
        }

    def tail_logs(self, *, lines: int = 200) -> str:
        if lines <= 0 or lines > 5000:
            raise NativeQdrantError(
                "Log line count must be between 1 and 5000.",
                category="invalid_log_line_count",
            )
        if not self.log_path.is_file():
            return "Qdrant log file does not exist yet."
        content = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])

    def _install_result(self, action: str, executable: Path) -> dict[str, Any]:
        return {
            "ok": True,
            "action": action,
            "version": QDRANT_VERSION,
            "executable": str(executable),
            "endpoint": QDRANT_HTTP_URL,
            "storage_preserved": True,
        }

    def _validated_executable(self) -> Path:
        if not self.receipt_path.is_file():
            raise NativeQdrantError(
                "Native Qdrant is not installed.", category="not_installed"
            )
        try:
            receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NativeQdrantError(
                "Qdrant install receipt is unreadable.", category="invalid_install_receipt"
            ) from exc
        if not isinstance(receipt, dict):
            raise NativeQdrantError(
                "Qdrant install receipt is invalid.", category="invalid_install_receipt"
            )
        relative_text = receipt.get("executable")
        if (
            receipt.get("version") != QDRANT_VERSION
            or receipt.get("archive_sha256") != QDRANT_ARCHIVE_SHA256
            or not isinstance(relative_text, str)
        ):
            raise NativeQdrantError(
                "Qdrant install receipt does not match the pinned release.",
                category="invalid_install_receipt",
            )
        relative = PurePosixPath(relative_text.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise NativeQdrantError(
                "Qdrant executable path in the receipt is unsafe.",
                category="invalid_install_receipt",
            )
        executable = self.install_dir.joinpath(*relative.parts).resolve(strict=False)
        try:
            executable.relative_to(self.install_dir.resolve(strict=False))
        except ValueError as exc:
            raise NativeQdrantError(
                "Qdrant executable path escapes the install directory.",
                category="invalid_install_receipt",
            ) from exc
        if not executable.is_file():
            raise NativeQdrantError(
                "Qdrant executable is missing.", category="executable_missing"
            )
        expected_executable_sha = receipt.get("executable_sha256")
        if not isinstance(expected_executable_sha, str) or (
            _sha256_file(executable) != expected_executable_sha
        ):
            raise NativeQdrantError(
                "Qdrant executable failed install-receipt verification.",
                category="executable_hash_mismatch",
            )
        return executable

    def _load_pid_record(self) -> dict[str, Any] | None:
        if not self.pid_path.is_file():
            return None
        try:
            record = json.loads(self.pid_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NativeQdrantError(
                "Qdrant PID record is unreadable; refusing to manage any process.",
                category="invalid_pid_record",
            ) from exc
        if (
            not isinstance(record, dict)
            or isinstance(record.get("pid"), bool)
            or not isinstance(record.get("pid"), int)
            or record["pid"] <= 0
            or not isinstance(record.get("executable"), str)
        ):
            raise NativeQdrantError(
                "Qdrant PID record is invalid; refusing to manage any process.",
                category="invalid_pid_record",
            )
        return record

    def _verified_process(self, record: dict[str, Any]) -> tuple[int | None, Path | None]:
        executable = self._validated_executable()
        recorded_executable = Path(str(record["executable"]))
        if not _same_path(recorded_executable, executable):
            raise NativeQdrantError(
                "PID record executable does not match the pinned installation.",
                category="pid_path_mismatch",
            )
        pid = int(record["pid"])
        try:
            actual = self.process_image(pid)
        except OSError as exc:
            raise NativeQdrantError(
                "Unable to verify the PID's process image; refusing to terminate it.",
                category="process_verification_failed",
            ) from exc
        if actual is None:
            return None, None
        if not _same_path(actual, executable):
            raise NativeQdrantError(
                "The PID belongs to another executable; refusing to terminate it.",
                category="pid_path_mismatch",
            )
        return pid, actual

    def _rotate_log_if_needed(self) -> None:
        if not self.log_path.is_file() or self.log_path.stat().st_size < _MAX_LOG_BYTES:
            return
        rotated = self.log_path.with_suffix(".log.1")
        rotated.unlink(missing_ok=True)
        os.replace(self.log_path, rotated)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install and manage pinned native Qdrant v1.15.4 on Windows",
    )
    parser.add_argument("action", choices=("install", "start", "stop", "restart", "status", "logs"))
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--storage-dir", type=Path)
    parser.add_argument("--force", action="store_true", help="Back up and replace an invalid install")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--tail-lines", type=int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manager = NativeQdrantManager(base_dir=args.base_dir, storage_dir=args.storage_dir)
        if args.action == "install":
            result = manager.install(force=args.force)
        elif args.action == "start":
            result = manager.start(timeout=args.timeout)
        elif args.action == "stop":
            result = manager.stop(timeout=min(args.timeout, 60.0))
        elif args.action == "restart":
            result = manager.restart(timeout=args.timeout)
        elif args.action == "logs":
            print(manager.tail_logs(lines=args.tail_lines))
            return 0
        else:
            result = manager.status()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if args.action == "status":
            return 0 if result.get("state") == "running" and result.get("healthy") else 1
        return 0
    except NativeQdrantError as exc:
        print(
            json.dumps(
                {"ok": False, "error_category": exc.category, "message": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
