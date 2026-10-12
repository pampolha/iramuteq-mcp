"""Job manager behaviour (no R needed): lifecycle, error kinds, prune, orphans."""

import json
import threading
import time

import pytest

from iramuteq_mcp.bridge import jobs, patches


def wait_for(manager, job_id, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = manager.get(job_id)
        if job.status not in jobs.ACTIVE:
            return job
        time.sleep(0.02)
    raise TimeoutError(job_id)


@pytest.fixture
def manager(tmp_path):
    return jobs.JobManager(str(tmp_path / "mcp-jobs"))


def test_done_job_keeps_result_and_output_dir(manager, tmp_path):
    out = str(tmp_path / "out")
    job = manager.submit("demo", {"x": 1}, lambda j: {"output_dir": out, "value": 42})
    done = wait_for(manager, job.id)
    assert done.status == "done"
    assert done.output_dir == out
    assert manager.result(job.id)["result"] == {"output_dir": out, "value": 42}


@pytest.mark.parametrize(
    "exc, kind",
    [
        (patches.RScriptError("boom", returncode=1, output="Error in foo"), "r_error"),
        (
            patches.RScriptError("boom", returncode=1, output="Error: cannot allocate vector of size 2 Gb"),
            "out_of_memory",
        ),
        (patches.RScriptError("killed", returncode=-9, output=""), "killed"),
        (ValueError("bad input"), "invalid_input"),
        (RuntimeError("bug"), "internal_error"),
    ],
)
def test_failures_are_classified(manager, exc, kind):
    def runner(job):
        raise exc

    job = wait_for(manager, manager.submit("demo", {}, runner).id)
    assert job.status == "error"
    assert job.error_kind == kind
    assert job.error


def test_cancel_before_start_never_runs_the_runner(manager):
    gate = threading.Event()
    ran = []

    def blocker(job):
        gate.wait(10)
        return {}

    def second(job):
        ran.append(job.id)
        return {}

    first = manager.submit("blocker", {}, blocker)
    queued = manager.submit("second", {}, second)
    manager.cancel(queued.id)
    gate.set()
    wait_for(manager, first.id)
    done = wait_for(manager, queued.id)
    assert done.status == "cancelled"
    assert done.error_kind == "cancelled"
    assert ran == []


def test_unknown_job_is_reported_as_none(manager):
    assert manager.get("nope") is None
    assert manager.result("nope") is None
    assert manager.cancel("nope") is None


def test_finished_jobs_are_kept_until_pruned(manager):
    job = wait_for(manager, manager.submit("demo", {}, lambda j: {}).id)
    time.sleep(0.05)
    assert [j["job_id"] for j in manager.list()] == [job.id]
    assert manager.result(job.id) is not None


def test_prune_removes_finished_only(manager, tmp_path):
    ok = wait_for(manager, manager.submit("ok", {}, lambda j: {}).id)
    bad = wait_for(manager, manager.submit("bad", {}, lambda j: 1 / 0).id)
    gate = threading.Event()
    running = manager.submit("running", {}, lambda j: gate.wait(10) or {})
    time.sleep(0.05)

    report = manager.prune()
    assert set(report["removed"]) == {ok.id, bad.id}
    assert report["skipped_active"] == [running.id]
    assert manager.get(running.id) is not None
    assert not (tmp_path / "mcp-jobs" / f"{ok.id}.json").exists()
    gate.set()
    wait_for(manager, running.id)


def test_prune_by_status_and_age(manager):
    ok = wait_for(manager, manager.submit("ok", {}, lambda j: {}).id)
    bad = wait_for(manager, manager.submit("bad", {}, lambda j: 1 / 0).id)

    # too recent for an age filter: nothing removed
    assert manager.prune(older_than_s=3600)["removed"] == []
    # status filter: only errors
    assert manager.prune(statuses=["error"])["removed"] == [bad.id]
    assert manager.get(ok.id) is not None


def test_orphaned_running_job_is_marked_interrupted(tmp_path):
    d = tmp_path / "mcp-jobs"
    d.mkdir()
    record = {
        "id": "abc123",
        "kind": "corpus_alceste",
        "status": "running",
        "params": {},
        "created": time.time(),
        "started": time.time(),
    }
    (d / "abc123.json").write_text(json.dumps(record))

    manager = jobs.JobManager(str(d))
    job = manager.get("abc123")
    assert job.status == "interrupted"
    assert "restarted" in job.error
    persisted = json.loads((d / "abc123.json").read_text())
    assert persisted["status"] == "interrupted"


def test_unreadable_record_is_skipped(tmp_path):
    d = tmp_path / "mcp-jobs"
    d.mkdir()
    (d / "broken.json").write_text("{not json")
    assert jobs.JobManager(str(d)).list() == []


def test_structurally_invalid_records_are_skipped(tmp_path):
    d = tmp_path / "mcp-jobs"
    d.mkdir()
    (d / "no-id.json").write_text(json.dumps({"kind": "demo", "status": "done"}))
    (d / "no-kind.json").write_text(json.dumps({"id": "abc123", "status": "done"}))
    (d / "array.json").write_text(json.dumps(["not", "a", "job"]))
    (d / "scalar.json").write_text("42")
    (d / "good.json").write_text(json.dumps({"id": "good1", "kind": "demo", "status": "done"}))

    manager = jobs.JobManager(str(d))
    assert [j["job_id"] for j in manager.list()] == ["good1"]


def test_analysis_lock_is_reentrant_and_exclusive():
    lock = jobs.analysis_lock()
    with lock:
        with lock:
            pass

    holding = threading.Event()
    release = threading.Event()

    def holder():
        with lock:
            holding.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    assert holding.wait(5)
    assert not lock.acquire(blocking=False)
    release.set()
    thread.join()
    assert lock.acquire(blocking=False)
    lock.release()


def test_classify_r_error_directly():
    assert jobs.classify_r_error(patches.RScriptError("x", returncode=-15)) == "killed"
    assert jobs.classify_r_error(patches.RScriptError("x", 1, "Cannot allocate vector")) == "out_of_memory"
    assert jobs.classify_r_error(patches.RScriptError("x", 1, "something else")) == "r_error"
