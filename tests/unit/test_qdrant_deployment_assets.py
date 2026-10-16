from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
COMPOSE_PATH = ROOT / "deploy" / "qdrant" / "compose.yaml"
MANAGER_PATH = ROOT / "scripts" / "manage_qdrant.ps1"
NATIVE_MANAGER_PATH = ROOT / "scripts" / "manage_qdrant_native.py"


def test_qdrant_compose_is_pinned_local_only_and_persistent() -> None:
    payload = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    service = payload["services"]["qdrant"]

    assert service["image"] == "qdrant/qdrant:v1.15.4"
    assert service["restart"] == "unless-stopped"
    assert service["ports"] == ["127.0.0.1:6333:6333", "127.0.0.1:6334:6334"]
    assert service["volumes"] == ["qdrant_storage:/qdrant/storage"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["environment"]["QDRANT__TELEMETRY_DISABLED"] == "true"
    assert payload["volumes"]["qdrant_storage"]["name"] == "modular_rag_qdrant_storage"
    assert service["healthcheck"]["retries"] >= 10


def test_qdrant_manager_preserves_storage_on_stop() -> None:
    script = MANAGER_PATH.read_text(encoding="utf-8")

    assert "Invoke-QdrantCompose stop" in script
    assert "Invoke-QdrantCompose down" not in script
    assert "down -v" not in script
    assert "http://127.0.0.1:6333/healthz" in script
    assert "WaitForExit(10000)" in script


def test_native_qdrant_manager_is_available_without_powershell_or_docker() -> None:
    script = NATIVE_MANAGER_PATH.read_text(encoding="utf-8")
    implementation = (ROOT / "src" / "production" / "native_qdrant.py").read_text(
        encoding="utf-8"
    )

    assert "src.production.native_qdrant import main" in script
    assert "qdrant-x86_64-pc-windows-msvc.zip" in implementation
    assert "QDRANT__SERVICE__HOST\": \"127.0.0.1" in implementation
    assert "QDRANT__TELEMETRY_DISABLED\": \"true" in implementation
    assert "QDRANT__STORAGE__STORAGE_PATH" in implementation
    assert "docker" not in script.lower()


def test_qdrant_client_is_pinned_to_the_native_server_compatibility_window() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"qdrant-client>=1.15.0,<1.16.0"' in project
