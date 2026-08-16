import json
from pathlib import Path

from scripts.resolve_protocol_binding import resolve_protocol_binding

ROOT = Path(__file__).resolve().parents[2]


def test_ci_bootstrap_restores_verified_protocol_and_backend_dependencies() -> None:
    script = (ROOT / "scripts/bootstrap-ci.ps1").read_text(encoding="utf-8")
    release_script = (ROOT / "scripts/build-release.ps1").read_text(encoding="utf-8")
    package_config = (ROOT / "packages/vibeocr-backend/pyproject.toml").read_text(
        encoding="utf-8"
    )
    lock = json.loads((ROOT / "release/protocol.lock.json").read_text(encoding="utf-8"))

    assert (
        resolve_protocol_binding(
            ROOT / "release/protocol.lock.json",
            ROOT / "packages/vibeocr-backend/pyproject.toml",
        )
        == "2.6.0"
    )

    assert "scripts/resolve_protocol_binding.py" in script
    assert "scripts/resolve_protocol_binding.py" in release_script
    assert '"vibeocr-runtime-contracts>=2.6.0,<3.0.0"' in package_config
    assert lock["version"] == "2.6.0"
    assert "vibeocr_runtime_client-2.6.0-py3-none-any.whl" in lock["artifacts"]
    assert "vibeocr_runtime_contracts-2.6.0-py3-none-any.whl" in lock["artifacts"]
    assert 'gh release download "v$protocolVersion"' in script
    assert 'gh release download "v$protocolVersion"' in release_script
    assert '"vibeocr_runtime_contracts-$protocolVersion-*.whl"' in script
    assert '"vibeocr_runtime_contracts-$protocolVersion-*.whl"' in release_script
    assert "gh release download" in script
    assert "gh attestation verify" in script
    assert "release/protocol.lock.json" in script
    assert "release/protocol.lock.json" in release_script
    assert "python -m pip install --group dev" in script
    assert "packages/vibeocr-backend" in script
