"""
visibility.py - per ray test of what the other video actually saw

Each 3D point of one video is projected into every camera of the other video
and its depth z is compared with the depth D that camera measured on the same
pixel:
    |D - z| <= tau  same     the surface is still there
    D - z  >  tau   free     the camera saw through the point: it is gone
    z - D  >  tau   occluded something nearer blocks it: no evidence
    out of image    out      not looked at: no evidence
A point only counts as changed after at least two views saw through it.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .scene import estimate_normals

# per-view codes
OUT, SAME, FREE, OCC, NODATA = 0, 1, 2, 3, 4
# per-point verdicts
KEPT, CHANGED, AMBIGUOUS, HIDDEN, UNSEEN = 1, 2, 3, 4, 5
VERDICT_NAMES = {KEPT: 'kept', CHANGED: 'changed', AMBIGUOUS: 'ambiguous',
                 HIDDEN: 'hidden', UNSEEN: 'unseen'}


class DepthCache:
    """Depth, 3x3 min and 3x3 max of each frame"""

    def __init__(self, scene):
        self.scene = scene
        self._c = {}

    def get(self, i):
        if i not in self._c:
            D = self.scene.depth[i]
            valid = np.isfinite(D)
            dmin = ndimage.minimum_filter(np.where(valid, D, np.inf), size=3, mode='nearest')
            dmax = ndimage.maximum_filter(np.where(valid, D, -np.inf), size=3, mode='nearest')
            self._c[i] = (np.where(valid, D, np.nan), dmin, dmax)
        return self._c[i]


def ray_check(points, scene, frames, tau_abs=0.06, tau_rel=0.04,
              margin=2, near=0.05, cache=None,
              normals=None):
    """Evidence code of every point in every frame"""
    cache = cache or DepthCache(scene)
    centers = scene.cam_centers()
    H, W = scene.hw
    out = np.full((len(frames), len(points)), OUT, np.uint8)
    for k, i in enumerate(frames):
        u, v, z = scene.project(points, i)
        ins = (z > near) & (u >= margin) & (u < W - margin) & (v >= margin) & (v < H - margin)
        if not ins.any():
            continue
        idx = np.nonzero(ins)[0]
        ui = np.round(u[idx]).astype(int).clip(0, W - 1)
        vi = np.round(v[idx]).astype(int).clip(0, H - 1)
        D, dmin, dmax = cache.get(i)
        zc = z[idx]
        tau = np.maximum(tau_abs, tau_rel * zc)
        if normals is not None:
            ray = points[idx] - centers[i]
            ray /= np.linalg.norm(ray, axis=1, keepdims=True) + 1e-9
            cos = np.abs(np.einsum('ij,ij->i', ray, normals[idx]))
            tau = tau / np.clip(cos, 0.3, 1.0)
        dc, mn, mx = D[vi, ui], dmin[vi, ui], dmax[vi, ui]
        code = np.full(len(idx), OCC, np.uint8)            # mixed / in front -> no evidence
        nodata = ~np.isfinite(mn)
        same = (np.abs(dc - zc) <= tau) | ((mn <= zc + tau) & (mx >= zc - tau))
        free = np.isfinite(mn) & (mn > zc + tau)
        code[same] = SAME
        code[free] = FREE
        code[nodata] = NODATA
        out[k, idx] = code
    return out


@dataclass
class Evidence:
    """Evidence of one point cloud judged by the other video"""
    n_same: np.ndarray
    n_free: np.ndarray
    n_in: np.ndarray          # views where the point was inside the image
    verdict: np.ndarray       # per-point KEPT / CHANGED / ...
    per_view: np.ndarray      # (F, M) codes, kept for evidence-frame selection
    frames: np.ndarray

    def counts(self, mask=None):
        v = self.verdict if mask is None else self.verdict[mask]
        return {name: int((v == code).sum()) for code, name in VERDICT_NAMES.items()}


def evaluate(points, scene, frames, cfg, cache=None, normals=None):
    """Verdict of every point: kept, changed, ambiguous, hidden or unseen"""
    if normals is None and len(points) >= 20:
        normals = estimate_normals(points)
    pv = ray_check(points, scene, frames, cfg.tau_abs, cfg.tau_rel, cache=cache, normals=normals)
    n_same = (pv == SAME).sum(0)
    n_free = (pv == FREE).sum(0)
    n_in = (pv != OUT).sum(0)
    tot = n_same + n_free
    ratio = np.where(tot > 0, n_free / np.maximum(tot, 1), 0)
    verdict = np.full(len(points), AMBIGUOUS, np.uint8)
    verdict[(tot > 0) & (ratio <= cfg.kept_ratio)] = KEPT
    verdict[(ratio >= cfg.changed_ratio) & (n_free >= cfg.min_free_views)] = CHANGED
    verdict[(tot == 0) & (n_in > 0)] = HIDDEN
    verdict[n_in == 0] = UNSEEN
    return Evidence(n_same, n_free, n_in, verdict, pv, np.asarray(frames))
