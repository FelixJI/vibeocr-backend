from __future__ import annotations

import json

import pytest
from test_runtime_installer import _fake_install, _release, _sha
from vibeocr.backend.runtime_control import RuntimeControl
from vibeocr.backend.runtime_install_plan import CAPABILITY, read_plan
from vibeocr.backend.runtime_installer import RuntimeInstaller, main
from vibeocr.backend.runtime_maintenance import (
    RuntimeInstallPlanStale,
    RuntimeOperationConflict,
)
from vibeocr.runtime_contracts.parser import parse_runtime_install_plan_response


def _control(tmp_path):
    manifest, component = _release(tmp_path / "release")
    raw = json.loads(manifest.read_text())
    raw["capabilities"].append(CAPABILITY)
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    binding = json.loads(component.read_text())
    binding["backend"]["runtime_manifest_sha256"] = _sha(manifest.read_bytes())
    component.write_text(json.dumps(binding), encoding="utf-8")
    calls = []

    def install(partial, manifest, profile):
        calls.append(profile)
        return _fake_install(partial, manifest, profile)

    def factory(**kwargs):
        kwargs.setdefault("accelerator", "cpu")
        return RuntimeInstaller(
            product_root=tmp_path / "product",
            component_lock=component,
            runtime_manifest=manifest,
            install_runner=kwargs.pop("install_runner", install),
            **kwargs,
        )

    return (
        RuntimeControl.from_installer_factory(factory),
        factory,
        calls,
        manifest,
        component,
    )


def test_preview_is_contract_valid_and_does_not_create_operation_or_runtime(tmp_path):
    control, _, calls, _, _ = _control(tmp_path)
    response = control.preview_install_plan(
        install_component_ids=("paddleocr-cpu",), required_capabilities=(CAPABILITY,)
    )
    parsed = parse_runtime_install_plan_response(response)
    assert set(parsed.plan.effective_component_ids) == {
        "rapidocr-base",
        "runtime_host",
        "paddleocr-cpu",
    }
    assert calls == []
    assert not (tmp_path / "product/runtime").exists()
    assert not (tmp_path / "product/state/operations").exists()
    assert parsed.plan.cost.download_bytes is None


def test_confirmation_and_expired_replay_keep_same_receipt(tmp_path):
    control, _, calls, _, _ = _control(tmp_path)
    plan = control.preview_install_plan(
        install_component_ids=("paddleocr-cpu",), required_capabilities=(CAPABILITY,)
    )["plan"]
    receipt = control.execute(
        operation="ensure",
        operation_id="op",
        plan_id=plan["plan_id"],
        required_capabilities=(CAPABILITY,),
    )
    record = read_plan(control.state_root, plan["plan_id"])
    record["plan"]["expires_at"] = "2000-01-01T00:00:00+00:00"
    (control.state_root / "install-plans" / f"{plan['plan_id']}.json").write_text(
        json.dumps(record)
    )
    replay = control.execute(
        operation="ensure",
        operation_id="op",
        plan_id=plan["plan_id"],
        required_capabilities=(CAPABILITY,),
    )
    assert replay == receipt
    assert receipt["snapshot"]["plan_id"] == plan["plan_id"]
    assert len(calls) == 1
    with pytest.raises(RuntimeOperationConflict):
        control.execute(
            operation="ensure",
            operation_id="new",
            plan_id=plan["plan_id"],
            required_capabilities=(CAPABILITY,),
        )


def test_marker_change_between_preview_and_confirmation_is_stale(tmp_path):
    control, _, calls, _, _ = _control(tmp_path)
    plan = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    control.execute(
        operation="ensure", operation_id="other", install_component_ids=("mineru-cpu",)
    )
    with pytest.raises(RuntimeInstallPlanStale):
        control.execute(
            operation="ensure",
            operation_id="stale",
            plan_id=plan["plan_id"],
            required_capabilities=(CAPABILITY,),
        )
    assert len(calls) == 1


def test_restart_and_device_switch_preserve_selected_engine(tmp_path):
    control, factory, calls, _, _ = _control(tmp_path)
    control.execute(
        operation="ensure", operation_id="cpu", install_component_ids=("paddleocr-cpu",)
    )
    factory().ensure()
    assert len(calls) == 1
    cuda = factory(accelerator="nvidia_cuda")
    assert set(cuda._desired_scope_ids()) == {
        "rapidocr-base",
        "runtime_host",
        "paddleocr-cuda",
    }
    cuda.ensure()
    assert (tmp_path / "product" / "runtime.rollback").is_dir()


def test_different_plan_cannot_reuse_operation_id(tmp_path):
    control, _, calls, _, _ = _control(tmp_path)
    first = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    control.execute(
        operation="ensure",
        operation_id="op",
        plan_id=first["plan_id"],
        required_capabilities=(CAPABILITY,),
    )
    second = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    with pytest.raises(RuntimeOperationConflict):
        control.execute(
            operation="ensure",
            operation_id="op",
            plan_id=second["plan_id"],
            required_capabilities=(CAPABILITY,),
        )
    assert len(calls) == 1


def test_host_preview_and_confirm_use_same_plan(tmp_path, monkeypatch, capsys):
    control, _, _, manifest, component = _control(tmp_path)
    monkeypatch.setattr(
        "vibeocr.backend.runtime_installer._runtime_control_from_request",
        lambda *a, **k: control,
    )
    request = {
        "protocol_version": 2,
        "request_kind": "install_plan",
        "product_root": str(tmp_path / "product"),
        "component_lock": str(component),
        "runtime_manifest": str(manifest),
        "install_component_ids": [],
        "required_capabilities": [CAPABILITY],
    }
    assert main(["--request-json", json.dumps(request)]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["response_kind"] == "install_plan"
    request.pop("install_component_ids")
    request.update(
        request_kind="start",
        operation="ensure",
        operation_id="host-confirm",
        plan_id=response["plan"]["plan_id"],
    )
    assert main(["--request-json", json.dumps(request)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_two_confirmations_cannot_install_the_same_baseline_twice(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    control, factory, calls, _, _ = _control(tmp_path)
    plan = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]

    def confirm(operation_id):
        other = RuntimeControl.from_installer_factory(factory)
        try:
            return other.execute(
                operation="ensure",
                operation_id=operation_id,
                plan_id=plan["plan_id"],
                required_capabilities=(CAPABILITY,),
            )["snapshot"]["operation_state"]
        except RuntimeOperationConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(confirm, ["first", "second"])) == [
            "conflict",
            "succeeded",
        ]
    assert len(calls) == 1


def test_failed_plan_retry_requires_fresh_plan_and_replays_command(tmp_path):
    from vibeocr.backend.runtime_installer import RuntimeInstallError

    control, factory, calls, _, _ = _control(tmp_path)
    preview = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]

    def fail(*args):
        raise RuntimeInstallError("candidate verification failed")

    control._installer_factory = lambda **kwargs: factory(install_runner=fail, **kwargs)
    with pytest.raises(RuntimeInstallError):
        control.execute(
            operation="ensure",
            operation_id="failed",
            plan_id=preview["plan_id"],
            required_capabilities=(CAPABILITY,),
        )
    control._installer_factory = factory
    with pytest.raises(ValueError, match="fresh preview"):
        control.command(
            command_id="no-plan",
            command="retry",
            target_operation_id="failed",
            new_operation_id="no-plan-op",
        )
    with pytest.raises(RuntimeOperationConflict):
        control.command(
            command_id="reused-plan",
            command="retry",
            target_operation_id="failed",
            new_operation_id="reused-plan-op",
            plan_id=preview["plan_id"],
            required_capabilities=(CAPABILITY,),
        )
    fresh = control.preview_install_plan(
        install_component_ids=("mineru-cpu",), required_capabilities=(CAPABILITY,)
    )["plan"]
    request = dict(
        command_id="retry",
        command="retry",
        target_operation_id="failed",
        new_operation_id="retry-op",
        plan_id=fresh["plan_id"],
        required_capabilities=(CAPABILITY,),
    )
    receipt = control.command(**request)
    assert receipt == control.command(**request)
    assert receipt["snapshot"]["operation_state"] == "succeeded"
    assert len(calls) == 1


def test_read_only_preview_reports_blocker_and_confirmation_does_not_install(
    tmp_path, monkeypatch
):
    from vibeocr.backend.runtime_maintenance import RuntimeInstallPlanBlocked

    control, _, calls, _, _ = _control(tmp_path)
    monkeypatch.setattr(
        RuntimeInstaller,
        "_installation_blockers",
        lambda self: [
            {"code": "insufficient_disk_space", "next_action": "free_disk_space"}
        ],
    )
    plan = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    with pytest.raises(RuntimeInstallPlanBlocked):
        control.execute(
            operation="ensure",
            operation_id="blocked",
            plan_id=plan["plan_id"],
            required_capabilities=(CAPABILITY,),
        )
    assert calls == []


@pytest.mark.parametrize(
    "version,blocked",
    [("527.00", True), ("528.33", False), ("610.88", False), ("unknown", True)],
)
def test_cuda_preview_checks_driver_compatibility_floor(
    tmp_path, monkeypatch, version, blocked
):
    from types import SimpleNamespace

    control, factory, _, _, _ = _control(tmp_path)
    installer = factory(
        accelerator="nvidia_cuda", install_component_ids=("paddleocr-cuda",)
    )
    installer._runner_reports_phases = True
    monkeypatch.setattr(
        "vibeocr.backend.runtime_installer.platform.machine", lambda: "AMD64"
    )
    monkeypatch.setattr(
        "vibeocr.backend.runtime_installer.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=version),
    )
    codes = [item["code"] for item in installer._installation_blockers()]
    assert any(code.startswith("nvidia_driver_") for code in codes) is blocked


def test_successful_noop_plan_cannot_be_confirmed_with_another_operation(tmp_path):
    control, _, calls, _, _ = _control(tmp_path)
    control.execute(operation="ensure", install_component_ids=())
    plan = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    receipt = control.execute(
        operation="ensure",
        operation_id="noop",
        plan_id=plan["plan_id"],
        required_capabilities=(CAPABILITY,),
    )
    assert receipt["snapshot"]["plan_id"] == plan["plan_id"]
    with pytest.raises(RuntimeOperationConflict):
        control.execute(
            operation="ensure",
            operation_id="other",
            plan_id=plan["plan_id"],
            required_capabilities=(CAPABILITY,),
        )
    assert len(calls) == 1
