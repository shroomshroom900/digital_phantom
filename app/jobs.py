"""
Asynchronous simulation jobs.

* One worker thread runs simulations, so the HTTP server stays responsive and
  solver state (caches, BLAS threads) is never shared between two concurrent runs.
* Each job holds an immutable SimConfig snapshot (pydantic frozen model).
* Every job gets a strictly increasing sequence number.  When a new job is
  submitted, jobs still waiting in the queue are marked 'superseded' and skipped,
  so an old request can never finish after (and overwrite) a newer one.
  The running job is allowed to finish; its result stays available, and the
  client keeps showing the newest COMPLETED result while the next one runs.
"""
import itertools
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from app import pipeline
from app.schemas import SimConfig

QUEUED, RUNNING, SUCCEEDED, FAILED, SUPERSEDED = "queued", "running", "succeeded", "failed", "superseded"


class Job:
    def __init__(self, seq, config: SimConfig, label=None):
        self.id = uuid.uuid4().hex[:12]
        self.seq, self.config, self.label = seq, config, label
        self.status, self.progress = QUEUED, "queued"
        self.submitted, self.started, self.finished = time.time(), None, None
        self.result, self.error = None, None

    def summary(self):
        return dict(id=self.id, seq=self.seq, label=self.label, status=self.status, progress=self.progress,
                    config_hash=self.config.config_hash(), submitted=self.submitted, started=self.started,
                    finished=self.finished, error=self.error)


class JobManager:
    def __init__(self, runner=pipeline.run, keep=40):
        self._runner = runner
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")
        self._jobs = OrderedDict()
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._keep = keep

    def submit(self, config: SimConfig, label=None, supersede=True):
        with self._lock:
            job = Job(next(self._seq), config, label)
            if supersede:
                for j in self._jobs.values():
                    if j.status == QUEUED:
                        j.status, j.progress, j.finished = SUPERSEDED, f"superseded by job {job.seq}", time.time()
            self._jobs[job.id] = job
            self._trim()
        self._pool.submit(self._execute, job)
        return job

    def _trim(self):
        while len(self._jobs) > self._keep:
            oldest = next(iter(self._jobs.values()))
            if oldest.status in (QUEUED, RUNNING):
                break
            self._jobs.popitem(last=False)

    def _execute(self, job: Job):
        with self._lock:
            if job.status != QUEUED:
                return
            job.status, job.started = RUNNING, time.time()

        def progress(msg):
            job.progress = msg
        try:
            res = self._runner(job.config, progress)
            res.update(job_id=job.id, seq=job.seq, label=job.label)
            job.result, job.status, job.progress = res, SUCCEEDED, "done"
        except pipeline.SimulationError as e:
            job.error = dict(message=str(e), stage=e.stage, details=e.details)
            job.status, job.progress = FAILED, "failed"
        except Exception as e:  # unexpected: still report something understandable
            job.error = dict(message=f"internal error: {type(e).__name__}: {e}", stage="internal",
                             details=dict(traceback=traceback.format_exc(limit=5)))
            job.status, job.progress = FAILED, "failed"
        finally:
            job.finished = time.time()

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def list(self):
        with self._lock:
            return [j.summary() for j in reversed(self._jobs.values())]

    def latest_seq(self):
        with self._lock:
            return max((j.seq for j in self._jobs.values()), default=0)

    def wait(self, job_id, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.get(job_id)
            if j is None or j.status in (SUCCEEDED, FAILED, SUPERSEDED):
                return j
            time.sleep(0.05)
        return self.get(job_id)
