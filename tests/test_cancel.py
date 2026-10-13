"""Cancelling a job while its R process is running (needs R, slow)."""

import time

import pytest
from helpers import run, write_large_corpus

from iramuteq_mcp import corpus_alceste, job_cancel, job_status
from iramuteq_mcp.bridge import jobs as jobs_mod

pytestmark = [pytest.mark.requires_r, pytest.mark.slow]


def _manager():
    from iramuteq_mcp import _job_manager

    return _job_manager()


def test_cancel_terminates_running_r_process(tmp_path):
    corpus = write_large_corpus(tmp_path / "large.txt", documents=400)
    job = run(corpus_alceste(str(corpus), lang="french", nbcl_p1=5, max_actives=2000))

    # wait until IRaMuTeQ has spawned its R process for this job
    record = _manager().get(job["job_id"])
    deadline = time.time() + 300
    while record.popen is None and record.status in jobs_mod.ACTIVE and time.time() < deadline:
        time.sleep(0.05)
    if record.popen is None:
        pytest.skip("analysis finished before an R process could be observed")

    process = record.popen  # keep a handle: the job clears it when it ends
    cancelled = run(job_cancel(job["job_id"]))
    assert cancelled["status"] in ("cancelled", "running")

    deadline = time.time() + 60
    while record.status in jobs_mod.ACTIVE and time.time() < deadline:
        time.sleep(0.1)
    assert record.status == "cancelled", record.error
    assert record.error_kind == "cancelled"
    assert process.poll() is not None, "R process still alive after cancel"
    assert process.returncode != 0, "R process should have been terminated, not finished"
    assert run(job_status(job["job_id"]))["status"] == "cancelled"
