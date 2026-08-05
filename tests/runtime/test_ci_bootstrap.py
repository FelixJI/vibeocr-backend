import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_ci_bootstrap_restores_verified_protocol_and_backend_dependencies() -> None:
    script = (ROOT / "scripts/bootstrap-ci.ps1").read_text(encoding="utf-8")
    release_script = (ROOT / "scripts/build-release.ps1").read_text(encoding="utf-8")
    package_config = (ROOT / "packages/vibeocr-backend/pyproject.toml").read_text(
        encoding="utf-8"
    )
    lock = json.loads((ROOT / "release/protocol.lock.json").read_text(encoding="utf-8"))

    assert "$protocolVersion = '2.2.0'" in script
    assert "$protocolVersion = '2.2.0'" in release_script
    assert '"vibeocr-runtime-contracts==2.2.0"' in package_config
    assert lock["version"] == "2.2.0"
    assert "gh release download" in script
    assert "gh attestation verify" in script
    assert "release/protocol.lock.json" in script
    assert "python -m pip install --group dev" in script
    assert "packages/vibeocr-backend" in script
