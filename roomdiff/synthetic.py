"""
synthetic.py - ray traced test room with known changes

chair     moved 1.8 m and turned 60 deg            -> moved
plant     taken off the table                       -> removed
backpack  put on the floor                          -> added
toy box   not in the detector vocabulary            -> added, unidentified object
rug       4 mm thick, removed                       -> looks different (colour only)
lamp      in a corner the 2nd video never films     -> can't tell
bin       behind a shelf in every 2nd video frame   -> can't tell
sofa, table, cabinet, shelf                         -> unchanged

Depth noise, per frame scale jitter, a small before/after misalignment, an
exposure change and an arbitrary global frame and scale are added, and the
detector misses some objects and gets one label wrong.
"""

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_CLASSES
from .objects import Detection, canonical
from .scene import AFTER, BEFORE, Scene

ROOM = (6.0, 5.0, 2.6)   # x, y, z extents (metres)


@dataclass
class Prim:
    """Box or cylinder in the synthetic room"""
    name: str
    label: str | None          # None -> the detector cannot name it
    kind: str                  # "box" | "cyl"
    xy: tuple
    size: tuple                # box: (w, d, h); cyl: (r, h)
    color: tuple
    z0: float = 0.0
    yaw: float = 0.0           # degrees
    identity: str = ''


def layout():
    """Objects before and after"""
    common = [
        Prim('sofa', 'sofa', 'box', (3.0, 4.55), (2.0, 0.8, 0.8), (52, 76, 140)),
        Prim('table', 'table', 'box', (3.0, 2.5), (1.2, 0.75, 0.74), (150, 105, 60)),
        Prim('cabinet', 'cabinet', 'box', (1.8, 0.3), (1.0, 0.45, 1.0), (205, 200, 185)),
        Prim('shelf', 'shelf', 'box', (4.2, 3.4), (0.3, 1.4, 1.7), (120, 90, 70)),
        Prim('bin', 'bin', 'cyl', (3.85, 3.4), (0.15, 0.4), (90, 95, 100)),
        Prim('lamp', 'lamp', 'cyl', (5.7, 4.7), (0.15, 1.6), (230, 200, 120)),
    ]
    before = common + [
        Prim('plant', 'plant', 'cyl', (3.3, 2.6), (0.13, 0.42), (60, 150, 70), z0=0.74),
        Prim('rug', None, 'box', (2.95, 3.6), (1.1, 0.6, 0.004), (150, 50, 140)),   # flat: geometry can't see it
        Prim('chair', 'chair', 'box', (3.9, 1.3), (0.62, 0.46, 0.9), (200, 60, 50), yaw=15),
    ]
    after = common + [
        Prim('chair', 'chair', 'box', (2.1, 1.2), (0.62, 0.46, 0.9), (200, 60, 50), yaw=75),
        Prim('backpack', 'backpack', 'box', (2.9, 1.6), (0.38, 0.26, 0.45), (95, 110, 50), yaw=20),
        Prim('toy_box', None, 'box', (3.4, 0.95), (0.34, 0.28, 0.24), (240, 170, 30), yaw=30),
    ]
    for p in before + after:
        p.identity = p.name
    return before, after


GROUND_TRUTH = {
    'moved': [('chair', (3.9, 1.3), (2.1, 1.2), 60.0)],
    'removed': [('plant', (3.3, 2.6))],
    'added': [('backpack', (2.9, 1.6)), ('unidentified object', (3.4, 0.95))],
    'unverified': [('lamp', (5.7, 4.7)), ('bin', (3.85, 3.4))],
    'appearance': [('rug', (2.95, 3.6), 'removed')],
    'unchanged': ['sofa', 'table', 'cabinet', 'shelf'],
}


def look_at(C, target, up=(0, 0, 1)):
    """World to camera pose looking at a target"""
    f = np.asarray(target, float) - C
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])                 # rows: camera x (right), y (down), z (fwd) in world
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ C
    return T


def trajectories(n=16, rng=None):
    """Camera paths of the two videos, from opposite sides of the room"""
    rng = rng or np.random.default_rng(0)
    s = np.linspace(0, 1, n)
    A, B = [], []
    for k in s:   # 'before': walking along the near wall, looking into the room (+x)
        C = np.array([0.6 + 0.5 * np.sin(np.pi * k), 0.9 + 3.3 * k, 1.38 + 0.04 * np.sin(5 * k)])
        tgt = np.array([4.8, 0.4 + 4.3 * k, 0.55])
        A.append(look_at(C, tgt))
    for k in s:   # 'after': from the far side, looking back (-x), different heights/angles
        C = np.array([5.3 - 0.25 * np.sin(np.pi * k), 2.7 - 1.8 * k, 1.48 + 0.05 * np.cos(4 * k)])
        tgt = np.array([1.0, 4.5 - 4.2 * k, 0.45])
        B.append(look_at(C, tgt))
    return A, B


def _ray_box(o, d, p):
    """Ray hit distance and normal for a box turned about z"""
    th = np.deg2rad(p.yaw)
    c, s = np.cos(-th), np.sin(-th)
    cx, cy = p.xy
    ox, oy = o[0] - cx, o[1] - cy
    lo_ = np.array([c * ox - s * oy, s * ox + c * oy, o[2]])
    ld = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]], 1)
    w, dd, h = p.size
    lo = np.array([-w / 2, -dd / 2, p.z0])
    hi = np.array([w / 2, dd / 2, p.z0 + h])
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (lo - lo_) / ld
        t2 = (hi - lo_) / ld
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    tn = np.nanmax(tmin, 1)
    tf = np.nanmin(tmax, 1)
    hit = (tf >= tn) & (tn > 1e-6)
    axis = np.nanargmax(tmin, 1)
    sign = -np.sign(ld[np.arange(len(ld)), axis])
    nl = np.zeros_like(ld)
    nl[np.arange(len(ld)), axis] = sign
    cw, sw = np.cos(th), np.sin(th)
    nw = np.stack([cw * nl[:, 0] - sw * nl[:, 1], sw * nl[:, 0] + cw * nl[:, 1], nl[:, 2]], 1)
    return np.where(hit, tn, np.inf), nw


def _ray_cyl(o, d, p):
    """Ray hit distance and normal for an upright cylinder"""
    r, h = p.size
    cx, cy = p.xy
    ox, oy = o[0] - cx, o[1] - cy
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (ox * d[:, 0] + oy * d[:, 1])
    c = ox ** 2 + oy ** 2 - r ** 2
    disc = b ** 2 - 4 * a * c
    with np.errstate(invalid='ignore', divide='ignore'):
        t = (-b - np.sqrt(disc)) / (2 * a)
    z = o[2] + t * d[:, 2]
    side = (disc >= 0) & (t > 1e-6) & (z >= p.z0) & (z <= p.z0 + h)
    t_side = np.where(side, t, np.inf)
    with np.errstate(invalid='ignore', divide='ignore'):
        tc = (p.z0 + h - o[2]) / d[:, 2]
    xc, yc = ox + tc * d[:, 0], oy + tc * d[:, 1]
    cap = (tc > 1e-6) & (xc ** 2 + yc ** 2 <= r ** 2)
    t_cap = np.where(cap, tc, np.inf)
    t_best = np.minimum(t_side, t_cap)
    hx, hy = ox + t_best * d[:, 0], oy + t_best * d[:, 1]
    n = np.stack([hx, hy, np.zeros_like(hx)], 1)
    with np.errstate(invalid='ignore'):
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    n[t_cap < t_side] = [0, 0, 1]
    return t_best, n


def _texture(P, scale=0.06):
    """Fixed texture noise from the world position"""
    k = np.floor(P / scale).astype(np.int64)
    h = (k[:, 0] * 73856093) ^ (k[:, 1] * 19349663) ^ (k[:, 2] * 83492791)
    return ((h % 1000) / 1000.0)


def render(prims, T_cw, K, H, W):
    """Depth, colour and instance id of one camera"""
    R, t = T_cw[:3, :3], T_cw[:3, 3]
    C = -R.T @ t
    v, u = np.mgrid[0:H, 0:W]
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
    d = dc @ R            # world direction with unit z-depth per unit t
    best = np.full(len(d), np.inf)
    inst = np.full(len(d), -1)
    normal = np.zeros_like(d)
    X, Y, Z = ROOM
    planes = [(2, 0.0, 0, (0, 0, 1)), (2, Z, 1, (0, 0, -1)), (0, 0.0, 2, (1, 0, 0)),
              (0, X, 3, (-1, 0, 0)), (1, 0.0, 4, (0, 1, 0)), (1, Y, 5, (0, -1, 0))]
    for ax, val, iid, n in planes:
        with np.errstate(divide='ignore', invalid='ignore'):
            tt = (val - C[ax]) / d[:, ax]
        ok = (tt > 1e-6) & (tt < best)
        best[ok], inst[ok] = tt[ok], iid
        normal[ok] = n
    for k, p in enumerate(prims):
        tt, nn = (_ray_box if p.kind == 'box' else _ray_cyl)(C, d, p)
        ok = tt < best
        best[ok], inst[ok] = tt[ok], 100 + k
        normal[ok] = nn[ok]
    P = C + d * best[:, None]
    base = np.zeros((len(d), 3))
    floor_tex = 0.75 + 0.25 * ((np.floor(P[:, 0] / 0.5) + np.floor(P[:, 1] / 0.5)) % 2)
    wall_cols = {0: (160, 130, 100), 1: (235, 235, 235), 2: (200, 205, 215), 3: (215, 200, 190),
                 4: (190, 210, 200), 5: (220, 215, 195)}
    for iid, col in wall_cols.items():
        base[inst == iid] = col
    base[inst == 0] *= floor_tex[inst == 0, None]
    for k, p in enumerate(prims):
        base[inst == 100 + k] = p.color
    light = np.array([0.4, 0.3, 0.85])
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.abs(normal @ light)
    tex = 0.85 + 0.15 * _texture(P)
    rgb = np.clip(base * shade[:, None] * tex[:, None], 0, 255).astype(np.uint8)
    return best.reshape(H, W).astype(np.float32), rgb.reshape(H, W, 3), inst.reshape(H, W)


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def small_rotation(deg, rng):
    """Rotation of deg degrees about a random axis"""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    th = np.deg2rad(deg)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def make_synthetic(n_frames=16, H=240, W=320, noise=True, seed=0, with_changes=True):
    """Scene, detections and ground truth of the synthetic room"""
    rng = np.random.default_rng(seed)
    before, after = layout()
    if not with_changes:
        after = before
    TA, TB = trajectories(n_frames, rng)
    K = np.array([[260.0, 0, W / 2], [0, 260.0, H / 2], [0, 0, 1]])
    imgs, depths, insts, Ts, sess, names = [], [], [], [], [], []
    for s, (prims, Tl) in enumerate([(before, TA), (after, TB)]):
        for k, T in enumerate(Tl):
            D, rgb, inst = render(prims, T, K, H, W)
            if noise and s == AFTER:   # the second video has a different exposure / white balance
                rgb = np.clip(rgb * np.array([1.07, 1.02, 0.94]) + 6, 0, 255).astype(np.uint8)
            if noise:
                D = D * (1 + rng.normal(0, 0.004, D.shape)) * (1 + rng.normal(0, 0.008))
            imgs.append(rgb), depths.append(D), insts.append(inst), Ts.append(T)
            sess.append(s), names.append(f"{'before' if s == 0 else 'after'}_{k:03d}")
    scene = Scene(images=np.stack(imgs), depth=np.stack(depths).astype(np.float32),
                  K=np.repeat(K[None], len(imgs), 0), T_cw=np.stack(Ts), session=np.array(sess),
                  frame_names=names, units='relative', meta={'source': 'synthetic'})
    gt_T_cw = scene.T_cw.copy()

    # detections simulated from instance masks
    dets = []
    synonyms = {'chair': ['chair', 'office chair'], 'plant': ['plant', 'potted plant']}
    for i in range(scene.n):
        prims = before if scene.session[i] == BEFORE else after
        inst = insts[i]
        for k, p in enumerate(prims):
            if p.label is None:
                continue
            m = inst == 100 + k
            if m.sum() < 120 or rng.random() < 0.15:     # missed detections
                continue
            v, u = np.nonzero(m)
            label = canonical(str(rng.choice(synonyms.get(p.label, [p.label]))), DEFAULT_CLASSES)
            if p.name == 'chair' and i == 3:
                label = 'stool'                            # one wrong label
            emb = _identity_vec(p.identity) + rng.normal(0, 0.12, 64)
            dets.append(Detection(frame=i, label=str(label), score=float(rng.uniform(0.45, 0.9)),
                                  box=(u.min(), v.min(), u.max(), v.max()), mask=m,
                                  embedding=emb / np.linalg.norm(emb)))

    if noise:
        # small before/after misregistration (joint reconstruction is not perfect)
        scene.apply_similarity(1.005, small_rotation(0.5, rng), np.array([0.03, -0.02, 0.01]),
                               frames=scene.frames_of(AFTER))
    # arbitrary global frame and scale (what a feed-forward reconstructor returns)
    scene.apply_similarity(0.37, random_rotation(rng), rng.normal(0, 1, 3))
    truth = GROUND_TRUTH if with_changes else {'moved': [], 'removed': [], 'added': [], 'unverified': [],
                                                'appearance': [], 'unchanged': ['sofa', 'table', 'cabinet', 'shelf']}
    gt = {'T_cw_room': gt_T_cw, 'layout': (before, after), 'truth': truth}
    return scene, dets, gt


def _identity_vec(name):
    """Fixed embedding per object, standing in for CLIP"""
    r = np.random.default_rng(sum(map(ord, name)) * 7919)
    v = r.normal(size=64)
    return v / np.linalg.norm(v)
