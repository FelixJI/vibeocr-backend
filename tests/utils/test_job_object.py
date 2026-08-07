from __future__ import annotations

import ctypes
from unittest.mock import Mock, call

import vibeocr.backend.utils.job_object as job_object_module
from vibeocr.backend.utils.job_object import JobObjectGuard


def test_job_object_guard_is_a_noop_outside_windows(monkeypatch) -> None:
    get_kernel32 = Mock()
    monkeypatch.setattr(job_object_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(job_object_module, "_get_kernel32", get_kernel32)

    with JobObjectGuard() as guard:
        assert guard.assign_from_popen(Mock(pid=1234)) is False
    guard.close()

    get_kernel32.assert_not_called()


def test_job_object_guard_assigns_process_and_closes_handles(monkeypatch) -> None:
    captured_flags: list[int] = []
    kernel32 = Mock()
    kernel32.CreateJobObjectW.return_value = 101
    kernel32.OpenProcess.return_value = 202
    kernel32.AssignProcessToJobObject.return_value = 1

    def capture_limits(
        _job_handle: int,
        _info_class: int,
        info_pointer,
        _info_size: int,
    ) -> int:
        info = ctypes.cast(
            info_pointer,
            ctypes.POINTER(job_object_module.JOBOBJECT_EXTENDED_LIMIT_INFORMATION),
        ).contents
        captured_flags.append(info.BasicLimitInformation.LimitFlags)
        return 1

    kernel32.SetInformationJobObject.side_effect = capture_limits
    monkeypatch.setattr(job_object_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(job_object_module, "_get_kernel32", lambda: kernel32)

    guard = JobObjectGuard(name="backend-test")
    assert guard.assign_from_popen(Mock(pid=4321)) is True
    guard.close()
    guard.close()

    assert captured_flags == [
        job_object_module.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        | job_object_module.JOB_OBJECT_LIMIT_BREAKAWAY_OK
    ]
    kernel32.AssignProcessToJobObject.assert_called_once_with(101, 202)
    assert kernel32.CloseHandle.call_args_list == [call(202), call(101)]


def test_job_object_guard_releases_handle_when_setup_fails(monkeypatch) -> None:
    kernel32 = Mock()
    kernel32.CreateJobObjectW.return_value = 101
    kernel32.SetInformationJobObject.return_value = 0
    monkeypatch.setattr(job_object_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(job_object_module, "_get_kernel32", lambda: kernel32)

    guard = JobObjectGuard()

    assert guard.assign_from_popen(Mock(pid=4321)) is False
    kernel32.CloseHandle.assert_called_once_with(101)
    kernel32.OpenProcess.assert_not_called()
