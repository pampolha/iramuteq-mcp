"""Job manager for long-running IRaMuTeQ analyses.

MCP tool calls are request/response; an ALCESTE run can take many minutes.
Heavy analyses therefore run in a background thread and return a job id
immediately; the agent polls via job_status / job_result / job_cancel.

The R child process is tracked through :mod:`patches` so a job can be
cancelled mid-run.  Job records are persisted under
``<UserConfigPath>/mcp-jobs/<id>.json`` so a restarted server can report
(orphaned) runs.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("iramuteq_mcp")

# Analyses are serialized: they share corpus sqlite handles and R is
# memory-hungry; parallel runs would thrash anyway.
_ANALYSIS_LOCK = threading.Lock()


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"  # queued | running | done | error | cancelled | interrupted
    phase: str = ""
    created: float = field(default_factory=time.time)
    started: float | None = None
    ended: float | None = None
    output_dir: str | None = None
    error: str | None = None
    result: dict | None = None
    cancel_requested: bool = False
    popen: Any = None
    thread: threading.Thread | None = None

    def summary(self) -> dict:
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "phase": self.phase,
            "created": self.created,
            "started": self.started,
            "ended": self.ended,
            "elapsed_s": round((self.ended or time.time()) - (self.started or self.created), 1),
            "output_dir": self.output_dir,
            "error": self.error,
            "params": self.params,
        }


class JobManager:
    def __init__(self, jobs_dir: str):
        self.jobs_dir = jobs_dir
        os.makedirs(jobs_dir, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._adopt_orphans()

    # ------------------------------------------------------------- lifecycle

    def _adopt_orphans(self) -> None:
        for name in sorted(os.listdir(self.jobs_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.jobs_dir, name), encoding="utf8") as f:
                    rec = json.load(f)
            except Exception:
                continue
            job = Job(id=rec["id"], kind=rec["kind"], params=rec.get("params", {}))
            for attr in ("status", "phase", "created", "started", "ended", "output_dir", "error"):
                if attr in rec:
                    setattr(job, attr, rec[attr])
            if job.status in ("queued", "running"):
                job.status = "interrupted"
                job.error = "MCP server restarted while the job was active"
                job.ended = time.time()
                self._persist(job)
            self._jobs[job.id] = job

    def _persist(self, job: Job) -> None:
        try:
            rec = {"id": job.id, **job.summary()}
            with open(os.path.join(self.jobs_dir, f"{job.id}.json"), "w", encoding="utf8") as f:
                json.dump(rec, f, indent=2, default=str)
        except Exception:
            log.exception("cannot persist job %s", job.id)

    # --------------------------------------------------------------- submit

    def submit(self, kind: str, params: dict, runner: Callable[[Job], dict]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
        with self._lock:
            self._jobs[job.id] = job
        self._persist(job)

        def _work():
            from . import patches

            job.status = "running"
            job.started = time.time()
            self._persist(job)
            patches.set_process_listener(lambda popen: setattr(job, "popen", popen))
            try:
                with _ANALYSIS_LOCK:
                    if job.cancel_requested:
                        raise InterruptedError("cancelled before start")
                    job.result = runner(job)
                if isinstance(job.result, dict) and job.result.get("output_dir"):
                    job.output_dir = job.result["output_dir"]
                job.status = "done"
            except InterruptedError as exc:
                job.status = "cancelled"
                job.error = str(exc)
            except patches.RScriptError as exc:
                job.status = "cancelled" if job.cancel_requested else "error"
                job.error = "%s\n%s" % (exc, exc.output[-4000:])
            except Exception as exc:  # noqa: BLE001 - report anything to the agent
                job.status = "cancelled" if job.cancel_requested else "error"
                job.error = "%s: %s" % (type(exc).__name__, exc)
                log.exception("job %s (%s) failed", job.id, kind)
            finally:
                job.ended = time.time()
                job.popen = None
                self._persist(job)

        job.thread = threading.Thread(target=_work, name=f"ira-job-{job.id}", daemon=True)
        job.thread.start()
        return job

    # ----------------------------------------------------------------- api

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        return [j.summary() for j in sorted(self._jobs.values(), key=lambda j: j.created)]

    def cancel(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown job {job_id}")
        job.cancel_requested = True
        popen = job.popen
        if popen is not None and popen.poll() is None:
            try:
                popen.terminate()
            except Exception:
                log.exception("terminate failed for job %s", job.id)
            deadline = time.time() + 5
            while popen.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            if popen.poll() is None:
                try:
                    popen.kill()
                except Exception:
                    pass
        self._persist(job)
        return job.summary()

    def result(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown job {job_id}")
        out = job.summary()
        out["result"] = job.result
        return out
