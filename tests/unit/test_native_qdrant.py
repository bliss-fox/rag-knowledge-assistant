from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.production import native_qdrant
from src.production.native_qdrant import NativeQdrantError, NativeQdrantManager


def _archive(path: Path, entries: dict[str, bytes] | None = None) -> str:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in (entries or {"qdrant.exe": b"pinned-binary"}).items():
            bundle.writestr(name, content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> NativeQdrantManager:
    archive = tmp_path / "release.zip"
    archive_sha = _archive(archive)
    monkeypatch.setattr(native_qdrant, "QDRANT_ARCHIVE_SHA256", archive_sha)

    def download(_url: str, destination: Path) -> None:
        shutil.copyfile(archive, destination)

    manager = NativeQdrantManager(
        base_dir=tmp_path / "qdrant",
        downloader=download,
        health_probe=lambda: False,
        port_in_use=lambda: False,
        platform_name="win32",
    )
    result = manager.install()
    assert result["action"] == "installed"
    return manager


def test_pinned_native_release_matches_verified_official_asset() -> None:
    assert native_qdrant.QDRANT_VERSION == "1.15.4"
    assert native_qdrant.QDRANT_ARCHIVE_URL.endswith(
        "/v1.15.4/qdrant-x86_64-pc-windows-msvc.zip"
    )
    assert native_qdrant.QDRANT_ARCHIVE_SHA256 == (
        "5348af719d5d9f3658597c8d3a8231bb8c38e7139a8076b69bd3dd59151d06d5"
    )


def test_install_verifies_archive_and_writes_immutable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _install(tmp_path, monkeypatch)

    receipt = json.loads(manager.receipt_path.read_text(encoding="utf-8"))
    executable = manager.install_dir / "qdrant.exe"

    assert executable.read_bytes() == b"pinned-binary"
    assert receipt["version"] == native_qdrant.QDRANT_VERSION
    assert receipt["archive_sha256"] == native_qdrant.QDRANT_ARCHIVE_SHA256
    assert receipt["executable_sha256"] == hashlib.sha256(b"pinned-binary").hexdigest()
    assert manager.install()["action"] == "already_installed"


def test_install_rejects_archive_hash_mismatch_without_replacing_install(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "release.zip"
    _archive(archive)

    def download(_url: str, destination: Path) -> None:
        shutil.copyfile(archive, destination)

    manager = NativeQdrantManager(
        base_dir=tmp_path / "qdrant",
        downloader=download,
        health_probe=lambda: False,
        platform_name="win32",
    )

    with pytest.raises(NativeQdrantError, match="SHA-256") as captured:
        manager.install()

    assert captured.value.category == "archive_hash_mismatch"
    assert not manager.install_dir.exists()


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.zip"
    _archive(archive, {"../qdrant.exe": b"not-safe"})

    with pytest.raises(NativeQdrantError) as captured:
        native_qdrant._safe_extract_archive(archive, tmp_path / "payload")

    assert captured.value.category == "unsafe_archive"
    assert not (tmp_path / "qdrant.exe").exists()


def test_start_uses_loopback_telemetry_off_and_persistent_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = _install(tmp_path, monkeypatch)
    health_results = iter((False, True))
    process = MagicMock(pid=4242)
    process.poll.return_value = None
    observed: dict[str, object] = {}

    def popen(command: list[str], **kwargs: object) -> MagicMock:
        observed["command"] = command
        observed.update(kwargs)
        return process

    times = iter((0.0, 0.0))
    manager = NativeQdrantManager(
        base_dir=installed.base_dir,
        health_probe=lambda: next(health_results),
        port_in_use=lambda: False,
        popen_factory=popen,
        monotonic=lambda: next(times),
        sleep=lambda _seconds: None,
        platform_name="win32",
    )

    result = manager.start(timeout=10)

    environment = observed["env"]
    assert isinstance(environment, dict)
    assert environment["QDRANT__SERVICE__HOST"] == "127.0.0.1"
    assert environment["QDRANT__SERVICE__HTTP_PORT"] == "6333"
    assert environment["QDRANT__SERVICE__GRPC_PORT"] == "6334"
    assert environment["QDRANT__TELEMETRY_DISABLED"] == "true"
    assert environment["QDRANT__STORAGE__STORAGE_PATH"] == str(manager.storage_dir)
    assert observed["command"] == [str(manager.install_dir / "qdrant.exe")]
    assert result["pid"] == 4242
    assert manager.storage_dir.is_dir()
    assert json.loads(manager.pid_path.read_text(encoding="utf-8"))["pid"] == 4242


def test_stop_refuses_to_kill_reused_pid_with_different_process_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _install(tmp_path, monkeypatch)
    manager.runtime_dir.mkdir(parents=True)
    manager.pid_path.write_text(
        json.dumps({"pid": 4242, "executable": str(manager.install_dir / "qdrant.exe")}),
        encoding="utf-8",
    )
    terminate = MagicMock()
    guarded = NativeQdrantManager(
        base_dir=manager.base_dir,
        process_image=lambda _pid: tmp_path / "another-program.exe",
        terminate_process=terminate,
        platform_name="win32",
    )

    with pytest.raises(NativeQdrantError) as captured:
        guarded.stop()

    assert captured.value.category == "pid_path_mismatch"
    terminate.assert_not_called()
    assert guarded.pid_path.exists()


def test_stop_terminates_only_verified_process_and_keeps_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = _install(tmp_path, monkeypatch)
    installed.storage_dir.mkdir(parents=True)
    sentinel = installed.storage_dir / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    installed.runtime_dir.mkdir(parents=True)
    executable = installed.install_dir / "qdrant.exe"
    installed.pid_path.write_text(
        json.dumps({"pid": 4242, "executable": str(executable)}), encoding="utf-8"
    )
    images = iter((executable, None))
    terminate = MagicMock()
    times = iter((0.0, 0.0))
    manager = NativeQdrantManager(
        base_dir=installed.base_dir,
        process_image=lambda _pid: next(images),
        terminate_process=terminate,
        monotonic=lambda: next(times),
        sleep=lambda _seconds: None,
        platform_name="win32",
    )

    result = manager.stop(timeout=5)

    terminate.assert_called_once_with(4242)
    assert result["storage_preserved"] is True
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not manager.pid_path.exists()


def test_stop_handles_windows_access_denied_during_process_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = _install(tmp_path, monkeypatch)
    installed.runtime_dir.mkdir(parents=True)
    executable = installed.install_dir / "qdrant.exe"
    installed.pid_path.write_text(
        json.dumps({"pid": 4242, "executable": str(executable)}), encoding="utf-8"
    )
    images = iter((executable, PermissionError(5, "access denied")))

    def process_image(_pid: int) -> Path | None:
        value = next(images)
        if isinstance(value, BaseException):
            raise value
        return value

    times = iter((0.0, 0.0))
    manager = NativeQdrantManager(
        base_dir=installed.base_dir,
        health_probe=lambda: False,
        port_in_use=lambda: False,
        process_image=process_image,
        terminate_process=lambda _pid: None,
        monotonic=lambda: next(times),
        sleep=lambda _seconds: None,
        platform_name="win32",
    )

    result = manager.stop(timeout=5)

    assert result["action"] == "stopped"
    assert not manager.pid_path.exists()


def test_tail_logs_is_bounded_and_returns_latest_lines(tmp_path: Path) -> None:
    manager = NativeQdrantManager(
        base_dir=tmp_path / "qdrant", platform_name="win32"
    )
    manager.logs_dir.mkdir(parents=True)
    manager.log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    assert manager.tail_logs(lines=2) == "two\nthree"
    with pytest.raises(NativeQdrantError) as captured:
        manager.tail_logs(lines=0)
    assert captured.value.category == "invalid_log_line_count"
