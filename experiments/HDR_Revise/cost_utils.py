#!/usr/bin/env python3
r"""Per-run computational-cost recorder (wall time, peak RAM, disk).

Attach ``RunCost(proc, out_dir, ...)`` immediately after ``subprocess.Popen``;
it samples the process-tree RSS in a daemon thread until the process exits, then
writes ``<out_dir>/cost.json`` AND appends one JSON line to ``cost_log``
(e.g. ``<RUN_ROOT>/costs.jsonl``). Call :func:`drain` before the orchestrator
exits to flush the last few runs. psutil is optional — without it wall-time and
disk are still recorded (peak_rss_mb = None).

Recorded fields: label, wall_s, peak_rss_mb, output_mb, returncode + any `extra`
(e.g. data_mb input size, n_hh, iter, optimizer). One line per ActivitySim run,
so the paper can report per-iteration compute cost for HDR / SPE / SPSA on both
Melbourne and VITM.
"""
import os, time, json, threading, atexit
try:
    import psutil
except Exception:                              # psutil missing -> RAM unavailable
    psutil = None

_ACTIVE = []


def _dir_size_mb(path):
    tot = 0
    try:
        for r, _, fs in os.walk(path):
            for f in fs:
                try:
                    tot += os.path.getsize(os.path.join(r, f))
                except OSError:
                    pass
    except OSError:
        pass
    return tot / 1e6


class RunCost(threading.Thread):
    """Samples a subprocess's peak RSS while it runs; finalizes on exit."""

    def __init__(self, proc, out_dir, label="", cost_log=None, interval=2.0, extra=None):
        super().__init__(daemon=True)
        self.proc, self.out_dir, self.label = proc, out_dir, label
        self.cost_log, self.interval, self.extra = cost_log, interval, (extra or {})
        self.peak_rss_mb, self.t0, self.rec = 0.0, time.time(), None
        _ACTIVE.append(self)
        self.start()

    def run(self):
        h = None
        if psutil:
            try:
                h = psutil.Process(self.proc.pid)
            except Exception:
                h = None
        while self.proc.poll() is None:
            if h:
                try:
                    rss = h.memory_info().rss
                    for c in h.children(recursive=True):
                        try:
                            rss += c.memory_info().rss
                        except Exception:
                            pass
                    if rss / 1e6 > self.peak_rss_mb:
                        self.peak_rss_mb = rss / 1e6
                except Exception:
                    pass
            time.sleep(self.interval)
        rec = {"label": self.label,
               "wall_s": round(time.time() - self.t0, 1),
               "peak_rss_mb": (round(self.peak_rss_mb, 1) if psutil else None),
               "output_mb": round(_dir_size_mb(self.out_dir), 1),
               "returncode": self.proc.returncode}
        rec.update(self.extra)
        self.rec = rec
        try:
            json.dump(rec, open(os.path.join(self.out_dir, "cost.json"), "w"), indent=2)
        except Exception:
            pass
        if self.cost_log:
            try:
                d = os.path.dirname(self.cost_log)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(self.cost_log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass


def attach(proc, out_dir, label="", cost_log=None, extra=None):
    """Convenience wrapper returning the RunCost (already started)."""
    return RunCost(proc, out_dir, label=label, cost_log=cost_log, extra=extra)


def drain(timeout=60):
    """Join all outstanding cost threads so the last runs are flushed."""
    for c in list(_ACTIVE):
        try:
            c.join(timeout=timeout)
        except Exception:
            pass


atexit.register(drain)                          # flush last runs on interpreter exit


def summarize(cost_log):
    """Read a costs.jsonl and return aggregate stats (for reporting)."""
    if not os.path.exists(cost_log):
        return None
    rows = []
    for line in open(cost_log):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return None
    walls = [r["wall_s"] for r in rows if r.get("wall_s") is not None]
    rams = [r["peak_rss_mb"] for r in rows if r.get("peak_rss_mb") is not None]
    outs = [r["output_mb"] for r in rows if r.get("output_mb") is not None]
    n = len(rows)
    return {
        "n_runs": n,
        "wall_s_mean": round(sum(walls) / len(walls), 1) if walls else None,
        "wall_s_total": round(sum(walls), 1) if walls else None,
        "peak_rss_mb_max": round(max(rams), 1) if rams else None,
        "peak_rss_mb_mean": round(sum(rams) / len(rams), 1) if rams else None,
        "output_mb_mean": round(sum(outs) / len(outs), 1) if outs else None,
    }
