"""Per-stage timing: live FPS + CSV log (your >=15 FPS evidence)."""
import csv
import time
from collections import deque
from contextlib import contextmanager
import numpy as np


class Timers:
    def __init__(self, csv_path=None):
        self.stage_ms = {}
        self.rows = []
        self.fps = 0.0
        self._ends = deque(maxlen=30)
        self._t0 = time.perf_counter()
        self.csv_path = csv_path
        self._n = 0

    def begin(self):
        self._t0 = time.perf_counter()
        self.stage_ms = {}

    @contextmanager
    def stage(self, name):
        s = time.perf_counter()
        try:
            yield
        finally:
            self.stage_ms[name] = (time.perf_counter() - s) * 1000

    def end_frame(self, t):
        now = time.perf_counter()
        self._ends.append(now)
        if len(self._ends) > 1:
            self.fps = (len(self._ends) - 1) / (self._ends[-1] - self._ends[0])
        total = (now - self._t0) * 1000
        proc = total - self.stage_ms.get("capture", 0.0)
        self.rows.append({"frame": self._n, "t": round(t, 3), **{k: round(v, 2) for k, v in self.stage_ms.items()},
                          "total_ms": round(total, 2), "processing_ms": round(proc, 2), "fps": round(self.fps, 1)})
        self._n += 1

    def close(self):
        if not self.rows:
            return
        if self.csv_path:
            keys = []
            for r in self.rows:
                keys += [k for k in r if k not in keys]
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.rows)
            print(f"[timing] wrote {self.csv_path}")
        proc = np.array([r["processing_ms"] for r in self.rows[10:]] or [r["processing_ms"] for r in self.rows])
        print(f"[timing] processing ms/frame: mean {proc.mean():.1f}, p95 {np.percentile(proc, 95):.1f}"
              f"  -> max sustainable ~{1000 / proc.mean():.0f} FPS (target >= 15)")
