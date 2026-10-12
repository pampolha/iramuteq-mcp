"""Job manager for long-running IRaMuTeQ analyses.

MCP tool calls are request/response; an ALCESTE run can take many minutes.
Heavy analyses therefore run in a background thread and return a job id
immediately; the client polls via job_status / job_result / job_cancel.

Lifecycle
---------
* A job is kept until it is cancelled or explicitly pruned (``prune``).
  Finished results are never discarded automatically.
* Each job record is persisted to ``<jobs_dir>/<id>.json`` so a restarted
  server can report jobs that were active when it stopped (``interrupted``).

Failures carry an ``error_kind``:
``invalid_input``, ``out_of_memory``, ``killed``, ``r_error``, ``internal_error``,
or ``cancelled`` (set together with status ``cancelled``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from . import patches

log = logging.getLogger("iramuteq_mcp")

FINISHED = ("done", "error", "cancelled", "interrupted")
ACTIVE = ("queued", "running")

# R reports allocation failures with these phrases (R >= 4.x and older builds).
_OOM_MARKERS = (
    "cannot allocate vector",
    "vector memory exhausted",
    "memory exhausted",
    "out of memory",
)

# Analyses are serialized: they share corpus sqlite handles and R is
# memory-hungry; parallel runs would thrash anyway.
_ANALYSIS_LOCK = threading.Lock()


def classify_r_error(exc: patches.RScriptError) -> str:
    """Map a failed R run to an error kind the client can act on."""
    if exc.returncode is not None and exc.returncode < 0:
        return "killed"  # terminated by a signal (e.g. OOM killer, cancel)
    text = (exc.output or "").lower()
    if any(marker in text for marker in _OOM_MARKERS):
        return "out_of_memory"
    return "r_error"


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
    error_kind: str | None = None
    result: dict | None = None
    cancel_requested: bool = False
    popen: Any = None
    thread: threading.Thread | None = None

    @property
    def elapsed(self) -> float:
        return round((self.ended or time.time()) - (self.started or self.created), 1)

    def summary(self) -> dict:
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "phase": self.phase,
            "created": self.created,
            "started": self.started,
            "ended": self.ended,
            "elapsed_s": self.elapsed,
            "output_dir": self.output_dir,
            "error": self.error,
            "error_kind": self.error_kind,
            "params": self.params,
        }


class JobManager:
    def __init__(self, jobs_dir: str):
        self.jobs_dir = jobs_dir
        os.makedirs(jobs_dir, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._adopt_orphans()

    # ------------------------------------------------------------ persistence

    def _record_path(self, job_id: str) -> str:
        return os.path.join(self.jobs_dir, f"{job_id}.json")

    def _persist(self, job: Job) -> None:
        try:
            rec = {"id": job.id, **job.summary()}
            with open(self._record_path(job.id), "w", encoding="utf8") as f:
                json.dump(rec, f, indent=2, default=str)
        except Exception:
            log.exception("cannot persist job %s", job.id)

    def _adopt_orphans(self) -> None:
        for name in sorted(os.listdir(self.jobs_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.jobs_dir, name), encoding="utf8") as f:
                    rec = json.load(f)
            except Exception:
                log.warning("skipping unreadable job record %s", name)
                continue
            job = Job(id=rec["id"], kind=rec["kind"], params=rec.get("params", {}))
            for attr in (
                "status",
                "phase",
                "created",
                "started",
                "ended",
                "output_dir",
                "error",
                "error_kind",
            ):
                if attr in rec:
                    setattr(job, attr, rec[attr])
            if job.status in ACTIVE:
                job.status = "interrupted"
                job.error = "MCP server restarted while the job was active"
                job.ended = time.time()
                self._persist(job)
            self._jobs[job.id] = job

    # ----------------------------------------------------------------- submit

    def submit(self, kind: str, params: dict, runner: Callable[[Job], dict]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
        with self._lock:
            self._jobs[job.id] = job
        self._persist(job)
        job.thread = threading.Thread(target=self._work, args=(job, runner), name=f"ira-job-{job.id}", daemon=True)
        job.thread.start()
        return job

    def _work(self, job: Job, runner: Callable[[Job], dict]) -> None:
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
            job.status, job.error_kind, job.error = "cancelled", "cancelled", str(exc)
        except patches.RScriptError as exc:
            if job.cancel_requested:
                job.status, job.error_kind = "cancelled", "cancelled"
            else:
                job.status, job.error_kind = "error", classify_r_error(exc)
            job.error = f"{exc}\n{(exc.output or '')[-4000:]}".rstrip()
        except ValueError as exc:
            job.status = "cancelled" if job.cancel_requested else "error"
            job.error_kind = "cancelled" if job.cancel_requested else "invalid_input"
            job.error = str(exc)
        except Exception as exc:
            job.status = "cancelled" if job.cancel_requested else "error"
            job.error_kind = "cancelled" if job.cancel_requested else "internal_error"
            job.error = f"{type(exc).__name__}: {exc}"
            log.exception("job %s (%s) failed", job.id, job.kind)
        finally:
            job.ended = time.time()
            job.popen = None
            patches.set_process_listener(None)
            self._persist(job)

    # --------------------------------------------------------------- queries

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        return [j.summary() for j in sorted(self._jobs.values(), key=lambda j: j.created)]

    def result(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return {**job.summary(), "result": job.result}

    # --------------------------------------------------------------- control

    def cancel(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status in FINISHED:
            return job.summary()
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
                    log.exception("kill failed for job %s", job.id)
        self._persist(job)
        return job.summary()

    def prune(
        self,
        job_ids: Iterable[str] | None = None,
        statuses: Iterable[str] = FINISHED,
        older_than_s: float | None = None,
    ) -> dict:
        """Delete finished job records. Active jobs are never pruned (cancel them first)."""
        wanted_status = set(statuses)
        now = time.time()
        removed, kept_active = [], []
        with self._lock:
            candidates = (
                [self._jobs[j] for j in job_ids if j in self._jobs]
                if job_ids is not None
                else list(self._jobs.values())
            )
            for job in candidates:
                if job.status in ACTIVE:
                    kept_active.append(job.id)
                    continue
                if job.status not in wanted_status:
                    continue
                if older_than_s is not None and (now - (job.ended or job.created)) < older_than_s:
                    continue
                del self._jobs[job.id]
                try:
                    os.remove(self._record_path(job.id))
                except FileNotFoundError:
                    pass
                removed.append(job.id)
        return {
            "removed": sorted(removed),
            "skipped_active": sorted(kept_active),
            "remaining": len(self._jobs),
        }
