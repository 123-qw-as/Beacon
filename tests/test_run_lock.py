import pytest

from math_agent.run_lock import RunLock, RunLockedError


def test_only_one_worker_can_hold_output_directory_lock(workdir):
    first = RunLock(workdir)
    second = RunLock(workdir)

    with first:
        with pytest.raises(RunLockedError):
            second.acquire()

    second.acquire()
    second.release()

