"""prediction/parallel.py

Run independent vo_rollout calls in parallel worker processes.

The full-horizon hypothesis rollouts never communicate with each other, so
they can run on separate CPU cores.  Python threads cannot do this (the GIL
lets only one thread run Python code at a time), so the work goes to a pool
of worker *processes*: each is a separate Python interpreter with its own
memory.  Inputs and results are copied between processes by pickling.

Typical usage
-------------
    pool = RolloutPool(n_workers=4)        # start the workers once, reuse them

    handles = [submit_rollout(pool, initial_states=..., dt=..., ...) for ...]
    trajs   = [h.result() for h in handles]  # waits; same order as submitted

With pool=None, submit_rollout runs vo_rollout immediately in this process,
so callers have one code path for both modes.  Results are identical: the
workers run the same deterministic code.

Notes
-----
- Create the pool once.  On Windows and macOS each worker is a fresh
  interpreter that re-imports numpy and this project (~1-2 s); a pool per
  call would cost more than it saves.
- Scripts that create a pool must keep the `if __name__ == "__main__":`
  guard: workers import the main script, and the guard stops them from
  running it.
- To debug inside vo_rollout, use pool=None (breakpoints and print do not
  work well inside workers).
"""

from __future__ import annotations

import os
import pickle
from concurrent.futures import Future, ProcessPoolExecutor, wait
from typing import Any, Callable, Optional

import numpy as np

from colregs_planner.vessel.state import VesselState

# vo_rollout is imported inside the functions below: vo_rollout.py itself
# imports this module, so a top-level import would be circular.


# ---------------------------------------------------------------------------
# Pending results
# ---------------------------------------------------------------------------

class Pending:
    """A result that may still be computing; result() waits for it.

    Wraps a collect() function that is called once, on the first result()
    call; its return value (or exception) is cached.
    """

    def __init__(self, collect: Callable[[], Any]) -> None:
        self._collect = collect
        self._done    = False
        self._value:  Any                     = None
        self._error:  Optional[BaseException] = None

    @classmethod
    def done(cls, value: Any) -> "Pending":
        """An already-available result."""
        return cls(lambda: value)

    def result(self) -> Any:
        if not self._done:
            try:
                self._value = self._collect()
            except BaseException as exc:      # cached and re-raised on every call
                self._error = exc
            self._done = True
        if self._error is not None:
            raise self._error
        return self._value


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------
# These functions run inside the worker processes, so they must be
# module-level (workers find them by import).

def _warm_up() -> None:
    """Import the rollout code (numpy, policies, ...) in the worker."""
    import colregs_planner.prediction.vo_rollout  # noqa: F401


def _run_rollout(payload: bytes) -> tuple[list[str], np.ndarray]:
    """Run vo_rollout and return the trajectories as one compact array.

    Returns (vessel_ids, array of shape (n_vessels, n_steps + 1, 5)) with
    columns north, east, psi, speed, r -- much cheaper to send back than
    lists of VesselState objects.  A ValueError from vo_rollout propagates
    to the caller's result().
    """
    from colregs_planner.prediction.vo_rollout import vo_rollout
    trajs = vo_rollout(**pickle.loads(payload))
    ids   = list(trajs)
    arr   = np.array([[(s.north, s.east, s.psi, s.speed, s.r) for s in trajs[vid]]
                      for vid in ids], dtype=float)
    return ids, arr


def _unpack(ids: list[str], arr: np.ndarray) -> dict[str, list[VesselState]]:
    return {
        vid: [VesselState(north=n, east=e, psi=p, speed=s, r=r) for n, e, p, s, r in rows]
        for vid, rows in zip(ids, arr.tolist())
    }


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

class RolloutPool:
    """Persistent pool of worker processes for vo_rollout calls.

    Parameters
    ----------
    n_workers : int
        Number of worker processes (>= 1).  More workers than parallel
        rollouts gains nothing; more than CPU cores slows things down.

    The workers are started (and have imported the project) when the
    constructor returns, so the first prediction does not pay the start-up
    cost.  They shut down at interpreter exit, or with shutdown().
    """

    def __init__(self, n_workers: int) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers
        self._executor = ProcessPoolExecutor(max_workers=n_workers)
        # One task per worker, submitted together, so every worker starts now.
        wait([self._executor.submit(_warm_up) for _ in range(n_workers)])

    def submit(self, **rollout_kwargs) -> Future:
        """Submit one vo_rollout(**rollout_kwargs) call; returns its Future.

        The arguments are pickled here, immediately, so later changes to
        them in this process cannot affect the rollout.
        """
        payload = pickle.dumps(rollout_kwargs, protocol=pickle.HIGHEST_PROTOCOL)
        return self._executor.submit(_run_rollout, payload)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "RolloutPool":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


def resolve_workers(setting: int | str, max_parallel: int | None) -> int:
    """Number of worker processes for a parallel_workers config value.

    setting      : an int (used as given; 0 = serial) or "auto".
    max_parallel : the most rollouts that can run at the same time, e.g.
                   min(top_k, 2^N) for one prediction; None = no limit (the
                   rollouts of many frames, as in run_AIS --metrics).

    "auto" uses one worker per parallel rollout, at most CPU cores - 1 (one
    core is left for the main process), and 0 when only one rollout runs at
    a time -- a pool would then only add overhead.
    """
    if setting != "auto":
        n = int(setting)
        if n < 0:
            raise ValueError(f"parallel_workers must be >= 0 or 'auto', got {setting!r}")
        return n
    cores = max(1, (os.cpu_count() or 1) - 1)
    n = cores if max_parallel is None else min(max_parallel, cores)
    return n if n > 1 else 0


def submit_rollout(pool: Optional[RolloutPool], **rollout_kwargs) -> Pending:
    """vo_rollout(**rollout_kwargs) on the pool, or right here if pool is None.

    Returns a Pending whose result() gives the trajectories dict (or raises
    the rollout's ValueError).  _timings is only honoured without a pool.
    """
    if pool is None:
        from colregs_planner.prediction.vo_rollout import vo_rollout
        try:
            value = vo_rollout(**rollout_kwargs)
        except ValueError as exc:
            err = exc
            def _raise():
                raise err
            return Pending(_raise)
        return Pending.done(value)

    rollout_kwargs.pop("_timings", None)
    future = pool.submit(**rollout_kwargs)
    return Pending(lambda: _unpack(*future.result()))
