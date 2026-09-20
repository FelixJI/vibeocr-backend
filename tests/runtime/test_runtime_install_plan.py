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
    assert (
        "native_model_preparation_not_estimated"
        in parsed.plan.cost.unknown_reason_codes
    )


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


def test_preview_persists_supervisor_blocker_until_fresh_preview(tmp_path):
    from vibeocr.backend.runtime_maintenance import RuntimeInstallPlanBlocked

    control, _, calls, _, _ = _control(tmp_path)
    blocker = {"code": "recognition_jobs_active", "next_action": "close_tasks"}
    plan = control.preview_install_plan(
        install_component_ids=(),
        required_capabilities=(CAPABILITY,),
        additional_blockers=(blocker,),
    )["plan"]
    assert plan["blockers"] == [blocker]
    with pytest.raises(RuntimeInstallPlanBlocked):
        control.execute(
            operation="ensure",
            operation_id="blocked-job",
            plan_id=plan["plan_id"],
            required_capabilities=(CAPABILITY,),
        )
    assert calls == []
    fresh = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    assert fresh["blockers"] == []
    control.execute(
        operation="ensure",
        operation_id="fresh-job",
        plan_id=fresh["plan_id"],
        required_capabilities=(CAPABILITY,),
    )
    assert len(calls) == 1


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_unsuccessful_confirmation_replays_receipt_after_expiry(tmp_path, state):
    from vibeocr.backend.runtime_installer import RuntimeInstallError
    from vibeocr.backend.runtime_maintenance import RuntimeOperationCancelled

    control, factory, calls, _, _ = _control(tmp_path)
    plan = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    failure = RuntimeInstallError if state == "failed" else RuntimeOperationCancelled

    def fail(partial, manifest, profile):
        if state == "cancelled":
            control._store.request_cancel("op")
            return _fake_install(partial, manifest, profile)
        raise failure("interrupted candidate")

    control._installer_factory = lambda **kwargs: factory(install_runner=fail, **kwargs)
    request = dict(
        operation="ensure",
        operation_id="op",
        plan_id=plan["plan_id"],
        required_capabilities=(CAPABILITY,),
    )
    with pytest.raises(failure):
        control.execute(**request)
    snapshot = control._store.snapshot("op")
    assert snapshot["operation_state"] == state
    record = read_plan(control.state_root, plan["plan_id"])
    record["plan"]["expires_at"] = "2000-01-01T00:00:00+00:00"
    (control.state_root / "install-plans" / f"{plan['plan_id']}.json").write_text(
        json.dumps(record)
    )
    restarted = RuntimeControl.from_installer_factory(factory)
    assert restarted.execute(**request)["snapshot"] == snapshot
    assert calls == []


def test_running_confirmation_replays_without_second_install(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    control, factory, calls, _, _ = _control(tmp_path)
    plan = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    entered, release = threading.Event(), threading.Event()

    def install(partial, manifest, profile):
        entered.set()
        assert release.wait(10)
        return _fake_install(partial, manifest, profile)

    control._installer_factory = lambda **kwargs: factory(
        install_runner=install, **kwargs
    )
    request = dict(
        operation="ensure",
        operation_id="op",
        plan_id=plan["plan_id"],
        required_capabilities=(CAPABILITY,),
    )
    other = RuntimeControl.from_installer_factory(factory)
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(control.execute, **request)
        assert entered.wait(5)
        try:
            replay = other.execute(**request)
            assert replay["snapshot"]["operation_state"] == "running"
            assert replay["snapshot"]["plan_id"] == plan["plan_id"]
            assert calls == []
        finally:
            release.set()
        assert pending.result()["snapshot"]["operation_state"] == "succeeded"


async def test_http_replays_confirm_and_retry_while_ocr_blocks_new_installs(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    from vibeocr.backend.runtime_installer import RuntimeInstallError
    from vibeocr.backend.supervisor.app import create_app
    from vibeocr.backend.supervisor.module import SupervisorModule, SupervisorOptions
    from vibeocr.runtime_contracts import JobKind, JobPriority

    control, factory, calls, _, _ = _control(tmp_path)
    plan = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]

    def fail(*args):
        raise RuntimeInstallError("candidate failed")

    control._installer_factory = lambda **kwargs: factory(install_runner=fail, **kwargs)
    with pytest.raises(RuntimeInstallError):
        control.execute(
            operation="ensure",
            operation_id="failed",
            plan_id=plan["plan_id"],
            required_capabilities=(CAPABILITY,),
        )
    control._installer_factory = factory
    fresh = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    retry = dict(
        command_id="retry",
        command="retry",
        target_operation_id="failed",
        new_operation_id="retried",
        plan_id=fresh["plan_id"],
        required_capabilities=[CAPABILITY],
    )
    with ThreadPoolExecutor() as executor:
        module = SupervisorModule(
            options=SupervisorOptions(instance_id="test-instance"),
            stager_root=tmp_path / "staging",
            executor=executor,
        )
        monkeypatch.setattr(module, "_dispatch", lambda *args: None)
        app = create_app(module, "test-token", runtime_control=control)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1",
            headers={"Authorization": "Bearer test-token"},
        ) as http:
            accepted = await http.post("/v2/runtime/maintenance/command", json=retry)
            assert accepted.status_code == 200, accepted.text
            preview = control.preview_install_plan(
                install_component_ids=(), required_capabilities=(CAPABILITY,)
            )["plan"]
            confirm = dict(
                operation="ensure",
                operation_id="confirmed",
                plan_id=preview["plan_id"],
                required_capabilities=[CAPABILITY],
            )
            original = await http.post("/v2/runtime/maintenance", json=confirm)
            assert original.status_code == 200, original.text
            new = control.preview_install_plan(
                install_component_ids=(), required_capabilities=(CAPABILITY,)
            )["plan"]
            module.submit(
                kind=JobKind.RECOGNITION,
                priority=JobPriority.INTERACTIVE,
                uploads=[("sample.png", "image/png", b"x")],
            )
            assert module.runtime_maintenance_blockers()
            replay = await http.post("/v2/runtime/maintenance", json=confirm)
            assert replay.status_code == 200, replay.text
            assert replay.json() == original.json()
            replay_retry = await http.post(
                "/v2/runtime/maintenance/command", json=retry
            )
            assert replay_retry.status_code == 200, replay_retry.text
            assert replay_retry.json() == accepted.json()
            blocked = await http.post(
                "/v2/runtime/maintenance",
                json={**confirm, "operation_id": "new", "plan_id": new["plan_id"]},
            )
            assert blocked.status_code == 423, blocked.text
            assert blocked.json()["code"] == "RUNTIME_BUSY"
            assert calls == ["win-x64-base"]


@pytest.mark.parametrize(
    "accelerators", [("cpu", "nvidia_cuda"), ("nvidia_cuda", "cpu")]
)
def test_base_device_switch_previews_candidate_cost_as_unknown(
    tmp_path, monkeypatch, accelerators
):
    control, factory, calls, _, _ = _control(tmp_path)
    monkeypatch.setattr(RuntimeInstaller, "_installation_blockers", lambda self: [])
    old, new = accelerators
    factory(accelerator=old, install_component_ids=()).ensure()
    plan = control.preview_install_plan(
        accelerator=new, install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    assert {item["action"] for item in plan["components"]} == {"replace"}
    assert plan["cost"]["download_bytes"] is None
    assert plan["cost"]["additional_disk_bytes"] is None
    assert "candidate_disk_usage_unknown" in plan["cost"]["unknown_reason_codes"]
    receipt = control.execute(
        operation="ensure",
        operation_id="switch",
        plan_id=plan["plan_id"],
        required_capabilities=(CAPABILITY,),
    )
    assert receipt["snapshot"]["operation_state"] == "succeeded"
    assert len(calls) == 2
    assert (tmp_path / "product/runtime.rollback").is_dir()


def test_confirmation_rechecks_failed_receipt_after_first_lookup_race(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from vibeocr.backend.runtime_installer import RuntimeInstallError

    control, factory, calls, _, _ = _control(tmp_path)
    plan = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    entered, release = threading.Event(), threading.Event()

    def delayed_factory(**kwargs):
        installer = factory(**kwargs)
        if kwargs.get("plan_id"):
            entered.set()
            assert release.wait(10)
        return installer

    def fail(*args):
        raise RuntimeInstallError("first confirmation failed")

    other = RuntimeControl.from_installer_factory(delayed_factory)
    control._installer_factory = lambda **kwargs: factory(install_runner=fail, **kwargs)
    request = dict(
        operation="ensure",
        operation_id="op",
        plan_id=plan["plan_id"],
        required_capabilities=(CAPABILITY,),
    )
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(other.execute, **request)
        assert entered.wait(5)
        try:
            with pytest.raises(RuntimeInstallError):
                control.execute(**request)
        finally:
            release.set()
        assert pending.result()["snapshot"] == control._store.snapshot("op")
    assert calls == []


def test_confirmation_rechecks_receipt_after_http_admission_race(tmp_path):
    from vibeocr.backend.runtime_installer import RuntimeInstallError
    from vibeocr.backend.runtime_lock import RuntimeLockTimeout

    control, factory, calls, _, _ = _control(tmp_path)
    plan = control.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    other = RuntimeControl.from_installer_factory(factory)
    request = dict(
        operation="ensure",
        operation_id="op",
        plan_id=plan["plan_id"],
        required_capabilities=(CAPABILITY,),
    )

    def fail(*args):
        raise RuntimeInstallError("first confirmation failed")

    other._installer_factory = lambda **kwargs: factory(install_runner=fail, **kwargs)

    def admission(action):
        # A competing HTTP request is accepted after this request's first lookup.
        with pytest.raises(RuntimeInstallError):
            other.execute(**request)
        raise RuntimeLockTimeout("another maintenance owns admission")

    receipt = control.execute(**request, run_maintenance=admission)
    assert receipt["snapshot"] == other._store.snapshot("op")
    assert receipt["snapshot"]["operation_state"] == "failed"
    assert calls == []


@pytest.mark.parametrize("devices", [("cpu", "nvidia_cuda"), ("nvidia_cuda", "cpu")])
def test_environment_control_follows_committed_device_after_switch(
    tmp_path, monkeypatch, devices
):
    _, factory, _, manifest, component = _control(tmp_path)
    original, switched = devices
    factory(accelerator=original, install_component_ids=()).ensure()
    monkeypatch.setenv("VIBEOCR_PRODUCT_ROOT", str(tmp_path / "product"))
    monkeypatch.setenv("VIBEOCR_COMPONENT_LOCK", str(component))
    monkeypatch.setenv("VIBEOCR_RUNTIME_MANIFEST", str(manifest))
    monkeypatch.setenv("VIBEOCR_RUNTIME_ACCELERATOR", original)
    control = RuntimeControl.from_environment()
    request = dict(install_component_ids=(), required_capabilities=(CAPABILITY,))
    assert control.preview_install_plan(**request)["plan"]["accelerator"] == original
    factory(accelerator=switched, install_component_ids=()).ensure()
    assert control.preview_install_plan(**request)["plan"]["accelerator"] == switched
    assert (
        RuntimeControl.from_environment().preview_install_plan(**request)["plan"][
            "accelerator"
        ]
        == switched
    )
    assert (
        control.preview_install_plan(accelerator=original, **request)["plan"][
            "accelerator"
        ]
        == original
    )
    factory(accelerator=original, install_component_ids=()).ensure()
    assert control.preview_install_plan(**request)["plan"]["accelerator"] == original


async def test_http_preview_distinguishes_inherited_and_explicit_download_sources(
    tmp_path,
):
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    from vibeocr.backend.supervisor.app import create_app
    from vibeocr.backend.supervisor.module import SupervisorModule, SupervisorOptions
    from vibeocr.runtime_contracts import SettingsSnapshot

    control, _, _, _, _ = _control(tmp_path)
    with ThreadPoolExecutor() as executor:
        module = SupervisorModule(
            options=SupervisorOptions(instance_id="source-test"),
            stager_root=tmp_path / "staging",
            executor=executor,
        )
        module.update_settings(SettingsSnapshot(download_source_ids=("pypi",)))
        app = create_app(module, "test-token", runtime_control=control)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1",
            headers={"Authorization": "Bearer test-token"},
        ) as http:
            request = {
                "required_capabilities": [CAPABILITY],
                "install_component_ids": [],
            }
            inherited = await http.post("/v2/runtime/install-plan", json=request)
            assert inherited.status_code == 200, inherited.text
            plan = inherited.json()["plan"]
            assert plan["requested_download_source_ids"] is None
            assert plan["effective_download_source_ids"] == ["pypi"]
            assert read_plan(control.state_root, plan["plan_id"])["plan"] == plan
            explicit = await http.post(
                "/v2/runtime/install-plan",
                json={**request, "download_source_ids": ["tuna-pypi"]},
            )
            assert explicit.status_code == 200, explicit.text
            assert explicit.json()["plan"]["requested_download_source_ids"] == [
                "tuna-pypi"
            ]
            assert explicit.json()["plan"]["effective_download_source_ids"] == [
                "tuna-pypi"
            ]
            receipt = await http.post(
                "/v2/runtime/maintenance",
                json={
                    "operation": "ensure",
                    "operation_id": "inherit-sources",
                    "plan_id": plan["plan_id"],
                    "required_capabilities": [CAPABILITY],
                },
            )
            assert receipt.status_code == 200, receipt.text
            snapshot = receipt.json()["snapshot"]
            assert snapshot.get("requested_download_source_ids") is None
            assert snapshot["effective_download_source_ids"] == ["pypi"]


def test_preview_holds_writer_lock_while_resolving_inherited_selection(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from vibeocr.backend.runtime_lock import RuntimeLockTimeout, RuntimeStoreLock

    control, factory, calls, _, _ = _control(tmp_path)
    factory(install_component_ids=("paddleocr-cpu",)).ensure()
    entered, release = threading.Event(), threading.Event()

    def delayed_factory(**kwargs):
        installer = factory(**kwargs)
        if CAPABILITY in kwargs.get("required_capabilities", ()):
            entered.set()
            assert release.wait(10)
        return installer

    preview_control = RuntimeControl.from_installer_factory(delayed_factory)
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(
            preview_control.preview_install_plan, required_capabilities=(CAPABILITY,)
        )
        assert entered.wait(5)
        try:
            with pytest.raises(RuntimeLockTimeout):
                with RuntimeStoreLock(
                    factory().paths.locks_root / "runtime-store.lock", timeout=0
                ):
                    pass
        finally:
            release.set()
        plan = pending.result()["plan"]
    assert "paddleocr-cpu" in plan["effective_component_ids"]
    factory(install_component_ids=("mineru-cpu",)).ensure()
    with pytest.raises(RuntimeInstallPlanStale):
        control.execute(
            operation="ensure",
            operation_id="old-preview",
            plan_id=plan["plan_id"],
            required_capabilities=(CAPABILITY,),
        )
    assert len(calls) == 2


@pytest.mark.parametrize("same_root", [False, True])
def test_shared_layout_plan_and_receipt_are_bound_to_initiating_product(
    tmp_path, same_root
):
    _, _, _, manifest, component = _control(tmp_path)
    bundle = tmp_path / "bundle"
    products = {"classic": "classic", "next": "classic" if same_root else "next"}
    for relative in set(products.values()):
        (bundle / relative).mkdir(parents=True)
    layout = bundle / "portable-layout.json"
    layout.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shared_root": "shared",
                "products": {key: {"root": value} for key, value in products.items()},
            }
        )
    )

    def control_for(product_id):
        def factory(**kwargs):
            kwargs.setdefault("accelerator", "cpu")
            return RuntimeInstaller(
                product_root=bundle / products[product_id],
                component_lock=component,
                runtime_manifest=manifest,
                layout_manifest=layout,
                product_id=product_id,
                install_runner=_fake_install,
                **kwargs,
            )

        return RuntimeControl.from_installer_factory(factory)

    owner, other = control_for("classic"), control_for("next")
    assert owner.state_root == other.state_root
    plan = owner.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    request = dict(
        operation="ensure",
        operation_id="owner-op",
        plan_id=plan["plan_id"],
        required_capabilities=(CAPABILITY,),
    )
    with pytest.raises(RuntimeInstallPlanStale):
        other.execute(**request)
    receipt = owner.execute(**request)
    with pytest.raises(RuntimeOperationConflict):
        other.execute(**request)
    assert owner.execute(**request) == receipt
    assert "product" not in plan


def _shared_factory(tmp_path):
    _, _, _, manifest, component = _control(tmp_path)
    bundle = tmp_path / "bundle"
    product = bundle / "classic"
    product.mkdir(parents=True)
    layout = bundle / "portable-layout.json"
    layout.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shared_root": "shared",
                "products": {"classic": {"root": "classic"}},
            }
        ),
        encoding="utf-8",
    )

    def factory(**kwargs):
        return RuntimeInstaller(
            product_root=product,
            component_lock=component,
            runtime_manifest=manifest,
            layout_manifest=layout,
            product_id="classic",
            install_runner=kwargs.pop("install_runner", _fake_install),
            **kwargs,
        )

    return factory


@pytest.mark.parametrize("failure", ["permission", "space"])
def test_shared_store_preflight_checks_actual_destination(
    tmp_path, monkeypatch, failure
):
    from types import SimpleNamespace

    import vibeocr.backend.runtime_installer as installer_module

    installer = _shared_factory(tmp_path)(accelerator="cpu", install_component_ids=())
    shared = installer.paths.store_root
    shared.mkdir(parents=True)
    installer._runner_reports_phases = True
    monkeypatch.setattr(installer_module.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        installer_module.os,
        "access",
        lambda path, mode: not (failure == "permission" and path == shared),
    )
    monkeypatch.setattr(
        installer_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(
            free=0 if failure == "space" and path == shared else 10**12
        ),
    )
    expected = (
        "runtime_not_writable" if failure == "permission" else "insufficient_disk_space"
    )
    assert expected in {entry["code"] for entry in installer._installation_blockers()}
    assert not installer.paths.runtime_root.exists()


async def test_host_environment_preserves_shared_http_plan_and_receipt(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    import vibeocr.backend.runtime_installer as installer_module
    from vibeocr.backend.supervisor.app import create_app
    from vibeocr.backend.supervisor.module import SupervisorModule, SupervisorOptions

    monkeypatch.setattr(
        installer_module,
        "_default_component_probe",
        lambda root, component_ids, profile: {item: True for item in component_ids},
    )
    factory = _shared_factory(tmp_path)
    launch = factory(accelerator="cpu", install_component_ids=()).ensure()
    for key, value in launch.environment.items():
        if key.startswith("VIBEOCR_"):
            monkeypatch.setenv(key, value)
    host = RuntimeControl.from_installer_factory(factory)
    plan = host.preview_install_plan(required_capabilities=(CAPABILITY,))["plan"]
    control = RuntimeControl.from_environment()
    assert control.state_root == host.state_root
    assert control._installer()._plan_baseline() == host._installer()._plan_baseline()
    with ThreadPoolExecutor() as executor:
        module = SupervisorModule(
            options=SupervisorOptions(instance_id="shared-test"),
            stager_root=tmp_path / "staging",
            executor=executor,
        )
        app = create_app(module, "test-token", runtime_control=control)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1",
            headers={"Authorization": "Bearer test-token"},
        ) as http:
            preview = await http.post(
                "/v2/runtime/install-plan", json={"required_capabilities": [CAPABILITY]}
            )
            assert preview.status_code == 200, preview.text
            assert (
                preview.json()["plan"]["effective_component_ids"]
                == plan["effective_component_ids"]
            )
            request = dict(
                operation="ensure",
                operation_id="shared-op",
                plan_id=plan["plan_id"],
                required_capabilities=[CAPABILITY],
            )
            response = await http.post("/v2/runtime/maintenance", json=request)
            assert response.status_code == 200, response.text
            assert response.json() == host.execute(
                **{**request, "required_capabilities": (CAPABILITY,)}
            )
            replay = await http.post("/v2/runtime/maintenance", json=request)
            assert replay.json() == response.json()
    assert not (tmp_path / "bundle/classic/runtime").exists()
    assert not (tmp_path / "bundle/classic/state/operations").exists()


def test_environment_rejects_missing_shared_layout_binding(tmp_path, monkeypatch):
    from vibeocr.backend.runtime_installer import RuntimeIdentityMismatch

    installer = _shared_factory(tmp_path)(accelerator="cpu", install_component_ids=())
    for key, value in installer._environment().items():
        if key.startswith("VIBEOCR_"):
            monkeypatch.setenv(key, value)
    monkeypatch.delenv("VIBEOCR_LAYOUT_MANIFEST", raising=False)
    monkeypatch.delenv("VIBEOCR_PRODUCT_ID", raising=False)
    with pytest.raises(RuntimeIdentityMismatch, match="store differs"):
        RuntimeControl.from_environment()


def test_shared_plan_retry_command_replay_checks_product_binding(tmp_path):
    from vibeocr.backend.runtime_installer import RuntimeInstallError
    from vibeocr.backend.runtime_maintenance import RuntimeCommandConflict

    factory = _shared_factory(tmp_path)
    installer = factory(accelerator="cpu")
    layout = tmp_path / "bundle/portable-layout.json"
    value = json.loads(layout.read_text(encoding="utf-8"))
    value["products"]["next"] = {"root": "classic"}
    layout.write_text(json.dumps(value), encoding="utf-8")
    owner = RuntimeControl.from_installer_factory(factory)
    other = RuntimeControl(
        product_root=installer.product_root,
        component_lock=installer.component_lock_path,
        runtime_manifest=installer.manifest.path,
        layout_manifest=layout,
        product_id="next",
    )
    assert other.state_root == owner.state_root
    plan = owner.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]

    def fail(*args):
        raise RuntimeInstallError("candidate failed")

    owner._installer_factory = lambda **kwargs: factory(install_runner=fail, **kwargs)
    with pytest.raises(RuntimeInstallError):
        owner.execute(
            operation="ensure",
            operation_id="failed",
            plan_id=plan["plan_id"],
            required_capabilities=(CAPABILITY,),
        )
    owner._installer_factory = factory
    fresh = owner.preview_install_plan(
        install_component_ids=(), required_capabilities=(CAPABILITY,)
    )["plan"]
    request = dict(
        command_id="retry",
        command="retry",
        target_operation_id="failed",
        new_operation_id="new-op",
        plan_id=fresh["plan_id"],
        required_capabilities=(CAPABILITY,),
    )
    receipt = owner.command(**request)
    assert receipt["snapshot"]["operation_state"] == "succeeded"
    with pytest.raises(RuntimeCommandConflict):
        other.command(**request)
    assert owner.command(**request) == receipt


@pytest.mark.parametrize("operation", ["ensure", "repair"])
@pytest.mark.parametrize(
    "failure", ["final_probe", "launch_directory", "success_event", "success_fsync"]
)
def test_activation_failure_restores_previous_runtime_and_choice(
    tmp_path, monkeypatch, operation, failure
):
    from pathlib import Path

    from vibeocr.backend.runtime_installer import RuntimeInstallError

    _, factory, _, _, _ = _control(tmp_path)
    original = factory(accelerator="cpu", install_component_ids=())
    original.ensure()
    marker = original.paths.runtime_root / ".installed.json"
    before = json.loads(marker.read_text(encoding="utf-8"))
    model = original.paths.state_root / "models" / "keep.txt"
    model.write_text("existing model", encoding="utf-8")
    installer = factory(
        operation_id="activation-failed",
        accelerator="nvidia_cuda" if operation == "ensure" else "cpu",
        install_component_ids=(),
    )

    def is_new():
        return (
            marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != before
        )

    if operation == "repair":
        monkeypatch.setattr(
            installer,
            "_drifted_component_ids",
            lambda **kwargs: [] if is_new() else ["rapidocr-base"],
        )

    def probe(root, components, profile):
        if (
            failure == "final_probe"
            and root == installer.paths.runtime_root
            and is_new()
        ):
            raise RuntimeInstallError("final-path probe failed")
        return {item: True for item in components}

    monkeypatch.setattr(installer, "_component_probe", probe)
    # repair's synthetic drift detector does not call component_probe; make the
    # final-path validation fail through the same detector for that branch.
    if operation == "repair" and failure == "final_probe":
        monkeypatch.setattr(
            installer, "_drifted_component_ids", lambda **kwargs: ["rapidocr-base"]
        )
    mkdir = Path.mkdir

    def prepare(path, *args, **kwargs):
        if failure == "launch_directory" and path == model.parent and is_new():
            raise PermissionError("launch directory denied")
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", prepare)
    append = installer._reporter._store.append

    def persist(operation_id, **kwargs):
        if (
            failure == "success_event"
            and kwargs["snapshot"]["operation_state"] == "succeeded"
        ):
            raise OSError("success event not written")
        return append(operation_id, **kwargs)

    monkeypatch.setattr(installer._reporter._store, "append", persist)
    if failure == "success_fsync":
        import vibeocr.backend.runtime_maintenance as maintenance

        fsync = maintenance.os.fsync
        journal = installer._reporter._store._events_path("activation-failed")
        injected = False

        def sync(fd):
            nonlocal injected
            if not injected and journal.exists():
                lines = journal.read_text(encoding="utf-8").splitlines()
                if (
                    lines
                    and json.loads(lines[-1])["snapshot"]["operation_state"]
                    == "succeeded"
                ):
                    injected = True
                    raise OSError("success journal fsync failed")
            return fsync(fd)

        monkeypatch.setattr(maintenance.os, "fsync", sync)
    with pytest.raises((RuntimeInstallError, OSError)):
        getattr(installer, operation)()
    assert json.loads(marker.read_text(encoding="utf-8")) == before
    assert model.read_text(encoding="utf-8") == "existing model"
    assert installer.maintenance_snapshot()["operation_state"] == "failed"
    restored = factory()
    assert restored.accelerator == "cpu"
    assert restored._launch().python_executable == original._launch().python_executable


@pytest.mark.parametrize("operation", ["ensure", "repair"])
@pytest.mark.parametrize("failure", ["metadata_projection", "event_delivery"])
def test_durable_success_survives_projection_or_delivery_failure(
    tmp_path, monkeypatch, operation, failure
):
    import vibeocr.backend.runtime_maintenance as maintenance

    _, factory, _, _, _ = _control(tmp_path)
    original = factory(accelerator="cpu", install_component_ids=())
    original.ensure()
    marker = original.paths.runtime_root / ".installed.json"
    before = json.loads(marker.read_text(encoding="utf-8"))
    installer = factory(
        operation_id="commit-test",
        accelerator="nvidia_cuda" if operation == "ensure" else "cpu",
        install_component_ids=(),
    )
    if operation == "repair":
        monkeypatch.setattr(
            installer,
            "_drifted_component_ids",
            lambda **kwargs: (
                ["rapidocr-base"]
                if json.loads(marker.read_text(encoding="utf-8")) == before
                else []
            ),
        )
    atomic = maintenance._atomic_json

    def project(path, value):
        if (
            failure == "metadata_projection"
            and path.name == "metadata.json"
            and (value.get("snapshot") or {}).get("operation_state") == "succeeded"
        ):
            raise OSError("metadata projection unavailable after journal fsync")
        return atomic(path, value)

    def deliver(event):
        if (
            failure == "event_delivery"
            and event["snapshot"]["operation_state"] == "succeeded"
        ):
            raise OSError("client disconnected after commit")

    monkeypatch.setattr(maintenance, "_atomic_json", project)
    monkeypatch.setattr(installer._reporter, "_event_sink", deliver)
    assert getattr(installer, operation)() is not None
    snapshot = installer.maintenance_snapshot()
    assert snapshot["operation_state"] == "succeeded"
    assert json.loads(marker.read_text(encoding="utf-8")) != before
    events = [
        json.loads(line)
        for line in installer._reporter._store._events_path("commit-test")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    states = [event["snapshot"]["operation_state"] for event in events]
    assert states.count("succeeded") == 1
    assert "failed" not in states
    monkeypatch.undo()
    store = maintenance.RuntimeOperationStore(installer.paths.state_root)
    assert store.snapshot("commit-test") == snapshot
    with pytest.raises(maintenance.RuntimeOperationError, match="terminal"):
        store.append(
            "commit-test",
            event_type="snapshot",
            snapshot={
                **snapshot,
                "sequence": snapshot["sequence"] + 1,
                "operation_state": "failed",
            },
            message_code="runtime.operation_failed",
        )
    assert store.snapshot("commit-test") == snapshot


@pytest.mark.parametrize("operation", ["ensure", "repair"])
@pytest.mark.parametrize(
    "failure, sequence", [("projection", 1), ("projection", 2), ("delivery", 1)]
)
def test_committed_event_failure_records_failed_and_allows_retry(
    tmp_path, monkeypatch, operation, failure, sequence
):
    import vibeocr.backend.runtime_maintenance as maintenance

    control, factory, _, _, _ = _control(tmp_path)
    original = factory(accelerator="cpu", install_component_ids=())
    original.ensure()
    marker = original.paths.runtime_root / ".installed.json"
    before = marker.read_bytes()
    installer = factory(
        operation_id="projection-failed",
        accelerator="cpu",
        install_component_ids=("paddleocr-cpu",) if operation == "ensure" else (),
    )
    if operation == "repair":
        monkeypatch.setattr(
            installer, "_drifted_component_ids", lambda **kwargs: ["rapidocr-base"]
        )
    atomic = maintenance._atomic_json
    injected = False

    def project(path, value):
        nonlocal injected
        snapshot = value.get("snapshot") or {}
        if (
            failure == "projection"
            and not injected
            and path.name == "metadata.json"
            and snapshot.get("operation_id") == "projection-failed"
            and snapshot.get("sequence") == sequence
        ):
            injected = True
            raise OSError("event metadata projection failed")
        return atomic(path, value)

    def deliver(event):
        nonlocal injected
        if failure == "delivery" and not injected and event["sequence"] == sequence:
            injected = True
            raise OSError("event delivery failed")

    monkeypatch.setattr(maintenance, "_atomic_json", project)
    monkeypatch.setattr(installer._reporter, "_event_sink", deliver)
    with pytest.raises(OSError, match="event .*failed"):
        getattr(installer, operation)()
    assert injected
    assert marker.read_bytes() == before
    snapshot = installer._reporter._store.snapshot("projection-failed")
    assert snapshot["operation_state"] == "failed"
    assert snapshot["sequence"] == sequence + 1
    monkeypatch.undo()
    receipt = control.command(
        command_id="retry-projection",
        command="retry",
        target_operation_id="projection-failed",
        new_operation_id="projection-retry",
    )
    assert receipt["snapshot"]["operation_state"] == "succeeded"
