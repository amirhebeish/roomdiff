"""
calibrate.py - measures the depth noise of this run and sets the tolerance from it

Inside one video nothing changes, so comparing its even frames with its odd
frames measures the depth noise of this phone, room and reconstruction. The
tolerance is set from it, and the same split is run through the change test as
a self-test: everything it flags there is a false alarm.
"""
import numpy as np

from .scene import BEFORE, AFTER
from .visibility import DepthCache, CHANGED, KEPT, evaluate


def depth_residuals(scene, points, frames, max_rel=0.25, cache=None):
    """Relative depth residual (D - z) / z of points seen by the given frames"""
    cache = cache or DepthCache(scene)
    H, W = scene.hw
    out = []
    for i in frames:
        u, v, z = scene.project(points, i)
        ok = (z > 0.05) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        if not ok.any():
            continue
        D = cache.get(i)[0][np.round(v[ok]).astype(int), np.round(u[ok]).astype(int)]
        r = (D - z[ok]) / z[ok]
        # occlusions and real changes give large residuals, keep the noise band only
        out.append(r[np.isfinite(r) & (np.abs(r) < max_rel)])
    return np.concatenate(out) if out else np.zeros(0)


def robust_sigma(x):
    """Standard deviation from the median absolute deviation"""
    if len(x) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def calibrate(scene, ps, cfg, sample=40000, seed=0):
    """Sets cfg.tau_rel from the depth noise measured inside each video, returns a report"""
    rng = np.random.default_rng(seed)
    cache = DepthCache(scene)
    r = []
    for s in (BEFORE, AFTER):
        # even frames against odd frames of the same video: nothing can have changed
        frames = scene.frames_of(s)
        even, odd = frames[0::2], frames[1::2]
        P = np.concatenate([scene.backproject(i, stride=3)[0] for i in even])
        if len(P) > sample:
            P = P[rng.choice(len(P), sample, replace=False)]
        r.append(depth_residuals(scene, P, odd, cache=cache))
    r = np.concatenate(r)
    sigma = robust_sigma(r)
    # 3.5 sigma keeps false alarms rare, the clamp stops a bad run from switching the test off
    tau = float(np.clip(cfg.tau_sigmas * sigma, cfg.tau_rel_min, cfg.tau_rel_max))
    report = {'depth_noise_rel': sigma, 'residuals_used': int(len(r)),
              'tau_rel_default': cfg.tau_rel, 'tau_rel': tau}
    cfg.tau_rel = tau
    return report


def self_test(scene, cfg):
    """Compares the first video with itself (even vs odd frames), where nothing changed"""
    frames = scene.frames_of(BEFORE)
    even, odd = frames[0::2], frames[1::2]
    if len(even) < 3 or len(odd) < 3:
        return {}
    pts = []
    for i in even:
        pts.append(scene.backproject(i, stride=2)[0])
    from .scene import voxel_downsample
    xyz, _ = voxel_downsample(np.concatenate(pts), None, cfg.voxel)
    ev = evaluate(xyz, scene, odd, cfg)
    kept = int((ev.verdict == KEPT).sum())
    changed = int((ev.verdict == CHANGED).sum())
    rate = changed / max(1, kept + changed)
    return {'false_alarm_rate': rate, 'points_tested': kept + changed}
