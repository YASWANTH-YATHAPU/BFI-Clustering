"""Measurement protocol.

Time    in-process, median over `repeats` runs of the complete algorithm (wall clock,
        time.perf_counter).  No memory sampler is running during timed runs.
Memory  peak *additional* resident memory of one run, measured in a FRESH process (spawn) so that
        earlier runs cannot pollute it: peak = max(RSS sampled every 1 ms, OS peak working set
        on Windows) minus the RSS just before the call (data, interpreter and imports excluded).
        It therefore includes every temporary (distance matrix, index, candidate lists, chunks).
All variants are single-threaded (n_jobs=1) so that they compete fairly with the numpy baseline.
"""
import gc
import threading
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def time_variant(name, X, c, repeats=3, **kw):
    """Median wall time of `repeats` runs and the result of the last run."""
    from bfi.variants import run_variant
    ts, res = [], None
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        res = run_variant(name, X, c, **kw)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), res


def _sample(stop, peak, proc):
    while not stop.is_set():
        peak[0] = max(peak[0], proc.memory_info().rss)
        time.sleep(0.001)


def _memory_worker(name, X, c, kw):
    import psutil
    from bfi.variants import run_variant
    run_variant(name, X[:50], min(c, 2), **dict(kw, K1=min(kw["K1"], 10)))   # warm-up imports
    gc.collect()
    proc = psutil.Process()
    mi = proc.memory_info()
    base, os_peak0 = mi.rss, getattr(mi, "peak_wset", None)
    peak = [base]
    stop = threading.Event()
    th = threading.Thread(target=_sample, args=(stop, peak, proc), daemon=True)
    th.start()
    t0 = time.perf_counter()
    run_variant(name, X, c, **kw)
    t = time.perf_counter() - t0
    stop.set()
    th.join()
    mi = proc.memory_info()
    best = peak[0]
    if os_peak0 is not None and mi.peak_wset > os_peak0:          # Windows: exact OS peak
        best = max(best, mi.peak_wset)
    return {"peak_mb": max(best - base, 0) / 2 ** 20, "time_mem_run": t}


def measure_memory(name, X, c, **kw):
    """Peak additional memory (MB) of one run of variant `name`, in a fresh process."""
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=ctx) as ex:
        return ex.submit(_memory_worker, name, np.ascontiguousarray(X), c, kw).result()
