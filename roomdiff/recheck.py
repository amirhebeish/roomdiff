"""
recheck.py - second opinion from the detector for things depth cannot see

Glass, clear plastic and very thin objects give no reliable depth: the camera
measures the wall behind a water bottle, so the free-space test says nothing
changed. For those the detector gets a vote. Every frame of the other video
that clearly looks at the object's spot (not blocked, not at the image edge)
is asked: is there still a thing with this name here? An object that its own
video detects almost every time it is in view, but that the other video never
finds in several clear looks, is gone.
"""

import cv2
import numpy as np

from .visibility import FREE, NODATA, SAME, DepthCache, ray_check


def frames_by_index(dets):
    """Detections grouped by frame"""
    by = {}
    for d in dets:
        by.setdefault(d.frame, []).append(d)
    return by


def _clear_of_blockers(points, scene, f, cache, cfg, win=15):
    """Points in view of frame f where no nearby measured surface is in front of them"""
    from scipy import ndimage
    key = ('near_min', f, win)
    if key not in cache._c:
        D = scene.depth[f]
        cache._c[key] = ndimage.minimum_filter(np.where(np.isfinite(D), D, np.inf), size=win, mode='nearest')
    dmin = cache._c[key]
    H, W = scene.hw
    u, v, z = scene.project(points, f)
    ins = (z > 0.05) & (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
    ok = np.zeros(len(points), bool)
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    tau = np.maximum(cfg.tau_abs, cfg.tau_rel * z)
    ok[ins] = dmin[vi[ins], ui[ins]] > z[ins] - tau[ins]
    return ok


def redetect(obj, scene, by_frame, frames, cfg, cache=None, max_points=1500):
    """Frames that clearly look at the object's spot, and those that detect it there"""
    cache = cache or DepthCache(scene)
    pts = obj.points
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    pv = ray_check(pts, scene, frames, cfg.tau_abs, cfg.tau_rel, cache=cache)
    H, W = scene.hw
    # any name the object was given often enough counts (a laptop is sometimes a "keyboard")
    top = max(obj.label_votes.values()) if obj.label_votes else 0
    names = {obj.label} | {k for k, v in obj.label_votes.items() if v >= 0.3 * top}
    looking, found = [], []
    for k, f in enumerate(frames):
        clear = (pv[k] == SAME) | (pv[k] == FREE)
        # glass often has no depth at all: there, nothing measured nearby in front of it
        # is enough to say the view is not blocked
        nodata = pv[k] == NODATA
        if nodata.any():
            clear[nodata] = _clear_of_blockers(pts[nodata], scene, f, cache, cfg)
        if clear.sum() < max(8, 0.5 * len(pts)):
            continue
        u, v, _ = scene.project(pts[clear], f)
        ui = np.clip(np.round(u).astype(int), 0, W - 1)
        vi = np.clip(np.round(v).astype(int), 0, H - 1)
        looking.append(int(f))
        cu, cv = int(np.median(ui)), int(np.median(vi))
        for d in by_frame.get(f, []):
            if d.label not in names:
                continue
            m = cv2.dilate(d.mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            if m[vi, ui].mean() >= 0.3 or m[cv, cu]:
                found.append(int(f))
                break
    return looking, found


def not_redetected(obj, scene, by_frame, cfg, cache=None):
    """Other-video frames that looked at the spot and found nothing, or None"""
    own = scene.frames_of(obj.session)
    other = scene.frames_of(1 - obj.session)
    l_own, f_own = redetect(obj, scene, by_frame, own, cfg, cache)
    if len(l_own) < cfg.recheck_min_looks or len(f_own) < cfg.recheck_own_rate * len(l_own):
        return None                       # the detector is not reliable on this object
    l_oth, f_oth = redetect(obj, scene, by_frame, other, cfg, cache)
    if len(l_oth) >= cfg.recheck_min_other_looks and not f_oth:
        return l_oth
    return None
