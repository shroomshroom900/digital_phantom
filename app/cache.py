"""
Thread-safe LRU caches for the expensive intermediate results.

Keys are sha256 digests of every input the value depends on PLUS the model
version (a hash of config.py, mesh.py, fem.py, fibre.py and the app version), so
editing the numerical code automatically invalidates everything.

    mesh   key = (model, geometry, mesh spacing)
    A      key = (mesh key, conductivity table)
    K row  key = (A key, electrode 3-D position, CG tolerance)

Adjoint rows are cached PER ELECTRODE: fem.adjoint_K solves A y = S_e^T one
electrode at a time and the solves are independent, so moving one electrode on
an unchanged mesh needs one CG solve, not n_e.  Rows are also kept on disk in
cache/app/ (safe to delete, like the rest of cache/).
"""
import hashlib
import json
import os
import threading
from collections import OrderedDict

import numpy as np

import config as C
from app import APP_VERSION

_CORE_FILES = ["config.py", "mesh.py", "fem.py", "fibre.py"]


def _model_version():
    h = hashlib.sha256(APP_VERSION.encode())
    for f in _CORE_FILES:
        with open(os.path.join(C.ROOT, f), "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


MODEL_VERSION = _model_version()


def make_key(*parts):
    s = json.dumps([MODEL_VERSION, *parts], sort_keys=True, default=float, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:24]


class LRU:
    """get_or_build(key, fn) -> (value, hit).  Builds happen outside the global
    lock but under a per-key lock, so two threads never build the same value."""

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._d = OrderedDict()
        self._lock = threading.Lock()
        self._key_locks = {}

    def peek(self, key):
        with self._lock:
            return key in self._d

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def get_or_build(self, key, fn):
        v = self.get(key)
        if v is not None:
            return v, True
        with self._lock:
            kl = self._key_locks.setdefault(key, threading.Lock())
        with kl:
            v = self.get(key)
            if v is not None:
                return v, True
            v = fn()
            self.put(key, v)
        with self._lock:
            self._key_locks.pop(key, None)
        return v, False

    def clear(self):
        with self._lock:
            self._d.clear()


class RowStore:
    """Adjoint rows K_e (one per electrode position) in memory + on disk."""

    def __init__(self, maxsize=64, disk_dir=os.path.join(C.CACHE_DIR, "app", "krows")):
        self.mem = LRU(maxsize)
        self.disk_dir = disk_dir
        os.makedirs(disk_dir, exist_ok=True)

    def _path(self, key):
        return os.path.join(self.disk_dir, f"{key}.npz")

    def has(self, key):
        return self.mem.peek(key) or os.path.exists(self._path(key))

    def get(self, key):
        v = self.mem.get(key)
        if v is not None:
            return v
        p = self._path(key)
        if os.path.exists(p):
            try:
                with np.load(p) as f:
                    v = (f["row"], int(f["iters"]), float(f["residual"]))
                self.mem.put(key, v)
                return v
            except Exception:          # corrupt / partial file: ignore and recompute
                return None
        return None

    def put(self, key, row, iters, residual):
        v = (row, int(iters), float(residual))
        self.mem.put(key, v)
        tmp = self._path(key) + ".tmp.npz"
        np.savez(tmp, row=row, iters=iters, residual=residual)
        os.replace(tmp, self._path(key))

    def clear(self, disk=False):
        self.mem.clear()
        if disk:
            for f in os.listdir(self.disk_dir):
                os.remove(os.path.join(self.disk_dir, f))


MESHES = LRU(3)
STIFFNESS = LRU(6)
SURFACES = LRU(6)
ROWS = RowStore()


def clear_all(disk=False):
    MESHES.clear(); STIFFNESS.clear(); SURFACES.clear(); ROWS.clear(disk=disk)
