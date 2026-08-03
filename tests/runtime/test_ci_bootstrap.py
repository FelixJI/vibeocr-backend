from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_ci_bootstrap_restores_verified_protocol_and_backend_dependencies() -> None:
    script = (ROOT / "scripts/bootstrap-ci.ps1").read_text(encoding="utf-8")

    assert "gh release download" in script
    assert "gh attestation verify" in script
    assert "release/protocol.lock.json" in script
    assert "python -m pip install --group dev" in script
    assert "packages/vibeocr-backend" in script
