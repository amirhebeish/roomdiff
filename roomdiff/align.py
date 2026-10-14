"""
align.py - gravity, floor, metric scale and before/after alignment

VGGT returns the scene in the first camera's frame at an unknown scale. Up
comes from the room itself (floor, ceiling and table tops face it, walls run
along it), the floor (RANSAC on planes facing up) gives z = 0, and the median
phone height above it gives metres. A damped point-to-plane ICP then nudges the 'after' cameras onto the
'before' ones, and is only kept if it improves the surface agreement.
"""

import numpy as np
from scipy.spatial import cKDTree

from .scene import AFTER, BEFORE, estimate_normals, session_points


def _fit_plane(p):
    """Least squares plane through points"""
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[2]
    return n, -n @ c


def find_horizontal_planes(xyz, up, thresh, max_angle_deg=25, n_planes=4, iters=400, rng=None):
    """Sequential RANSAC for planes close to horizontal"""
    rng = rng or np.random.default_rng(0)
    cos_max = np.cos(np.deg2rad(max_angle_deg))
    remaining = xyz
    planes = []
    for _ in range(n_planes):
        if len(remaining) < 50:
            break
        idx = rng.integers(0, len(remaining), size=(iters, 3))
        p0, p1, p2 = remaining[idx[:, 0]], remaining[idx[:, 1]], remaining[idx[:, 2]]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n, axis=1)
        ok = nn > 1e-12
        n = n[ok] / nn[ok, None]
        p0 = p0[ok]
        horiz = np.abs(n @ up) >= cos_max
        n, p0 = n[horiz], p0[horiz]
        if len(n) == 0:
            break
        d = -np.einsum('ij,ij->i', n, p0)
        sub = remaining if len(remaining) <= 8000 else remaining[rng.choice(len(remaining), 8000, replace=False)]
        counts = (np.abs(sub @ n.T + d) < thresh).sum(0)
        k = int(np.argmax(counts))
        nk, dk = n[k], d[k]
        inl = np.abs(remaining @ nk + dk) < thresh
        nk, dk = _fit_plane(remaining[inl])
        if nk @ up < 0:
            nk, dk = -nk, -dk
        inl = np.abs(remaining @ nk + dk) < thresh
        planes.append({'n': nk, 'd': dk, 'count': int(inl.sum()),
                       'height': float(np.median(remaining[inl] @ up))})
        remaining = remaining[~inl]
    return planes


def manhattan_up(P, prior, max_dev_deg=40, iters=4):
    """Up direction from the room itself: floor, ceiling and table tops face it, walls are parallel to it"""
    n, planarity = estimate_normals(P, k=16, return_planarity=True)
    n = n[planarity > 0.5]
    if len(n) < 500:
        return prior, 0.0
    # coarse search over directions near the prior (people tilt the phone down while filming,
    # so the camera up vector alone can be off by 20 degrees)
    g = np.arange(4000) + 0.5
    phi, th = np.arccos(1 - 2 * g / 4000), np.pi * (1 + 5 ** 0.5) * g
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    dirs = dirs[dirs @ prior >= np.cos(np.deg2rad(max_dev_deg))]
    sub = n if len(n) <= 20000 else n[np.random.default_rng(0).choice(len(n), 20000, replace=False)]
    c = np.abs(sub @ dirs.T)
    score = (c > np.cos(np.deg2rad(6))).sum(0) + (c < np.sin(np.deg2rad(6))).sum(0)
    up = dirs[np.argmax(score)]
    # refine: horizontal surfaces parallel to up, walls perpendicular to it
    for _ in range(iters):
        c = np.abs(n @ up)
        H, V = n[c > np.cos(np.deg2rad(10))], n[c < np.sin(np.deg2rad(10))]
        if len(H) + len(V) < 100:
            break
        M = V.T @ V / max(len(V), 1) - H.T @ H / max(len(H), 1)
        w, v = np.linalg.eigh(M)
        u = v[:, 0]
        up = u if u @ up > 0 else -u
    c = np.abs(n @ up)
    support = float(((c > np.cos(np.deg2rad(6))) | (c < np.sin(np.deg2rad(6)))).mean())
    return up, support


def gravity_align(scene, camera_height_m=1.4, rng=None):
    """Moves the scene so the floor is z = 0, z is up and units are metres"""
    # 1. gravity prior from camera orientations (image "up" is -y in OpenCV)
    ups = np.stack([-scene.T_cw[i, 1, :3] for i in range(scene.n)])
    prior = ups.mean(0)
    prior /= np.linalg.norm(prior)

    allp = np.concatenate([scene.backproject(i, stride=3)[0] for i in range(scene.n)])
    rng = rng or np.random.default_rng(0)
    if len(allp) > 60000:
        allp = allp[rng.choice(len(allp), 60000, replace=False)]
    centers = scene.cam_centers()
    scale_hint = float(np.median(np.linalg.norm(allp - centers.mean(0), axis=1)))
    thresh = 0.012 * scale_hint

    # the room's own geometry is a better level than the phone
    up, manhattan_support = manhattan_up(allp, prior)
    planes = find_horizontal_planes(allp, up, thresh, max_angle_deg=6, n_planes=8, rng=rng)
    floor = None
    if planes:
        big = max(p['count'] for p in planes)
        cands = [p for p in planes if p['count'] >= max(0.2 * big, 0.02 * len(allp))]
        # the floor: lowest well-supported horizontal plane with the cameras above it
        cands = [p for p in cands if np.median(centers @ p['n'] + p['d']) > 0]
        if cands:
            floor = min(cands, key=lambda p: p['height'])

    info = {'floor_found': floor is not None}
    if floor is not None:
        n, d = floor['n'], floor['d']
    else:   # fall back: gravity prior + a low percentile of point heights
        n, d = up, -np.percentile(allp @ up, 2)
    cam_h = centers @ n + d
    med_h = float(np.median(cam_h))
    scale = camera_height_m / med_h if med_h > 1e-9 else 1.0

    # 2. build the room frame
    c0 = centers[scene.frames_of(BEFORE)[0]]
    origin = c0 - (c0 @ n + d) * n
    fwd = scene.T_cw[scene.frames_of(BEFORE)[0], 2, :3]
    x = fwd - (fwd @ n) * n
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(n, [1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    Ra = np.stack([x, y, n])               # rows = new axes in old coords
    t = -scale * Ra @ origin
    scene.apply_similarity(scale, Ra, t)
    scene.units = 'm'
    info.update({'scale': float(scale), 'camera_height_raw': med_h,
                 'up_prior_vs_floor_deg': float(np.degrees(np.arccos(np.clip(prior @ n, -1, 1)))),
                 'manhattan_support': manhattan_support,
                 'floor_inliers': int(floor['count']) if floor else 0})
    return info


def umeyama(src, dst, with_scale=True):
    """Similarity transform (scale, rotation, translation) between matched points"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    c = float(np.trace(np.diag(S) @ E) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - c * R @ mu_s
    return c, R, t


def _rotvec(w):
    """Rotation matrix from a rotation vector"""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def refine_after_onto_before(scene, voxel=0.03, iters=40, max_rot_deg=6.0, max_trans=0.4,
                             max_scale_dev=0.03):
    """Point-to-plane Sim(3) ICP of the after frames onto the before frames"""
    A = session_points(scene, BEFORE, voxel).xyz.astype(np.float64)
    B = session_points(scene, AFTER, voxel).xyz.astype(np.float64)
    if len(A) < 200 or len(B) < 200:
        return {'applied': False, 'reason': 'too few points'}
    nA, plA = estimate_normals(A, return_planarity=True)
    nB, plB = estimate_normals(B, return_planarity=True)
    goodA = plA > 0.3
    tree = cKDTree(A)

    def agreement(P, N, gate=0.04):
        d, j = tree.query(P, distance_upper_bound=gate)
        m = np.isfinite(d)
        jj = np.where(m, j, 0)
        m &= np.abs(np.einsum('ij,ij->i', N, nA[jj])) > 0.85
        return float(m.mean())

    cen = B.mean(0)
    before = agreement(B, nB)
    R_tot, t_tot, c_tot = np.eye(3), np.zeros(3), 1.0
    for gate in np.geomspace(0.25, 0.04, iters):
        P = c_tot * (B - cen) @ R_tot.T + cen + t_tot
        N = nB @ R_tot.T
        d, j = tree.query(P, distance_upper_bound=gate)
        m = np.isfinite(d)
        jj = np.where(m, j, 0)
        n = nA[jj]
        m &= goodA[jj] & (np.abs(np.einsum('ij,ij->i', N, n)) > 0.85)
        if m.sum() < 100:
            break
        p, q, n = P[m], A[jj[m]], n[m]
        r = np.einsum('ij,ij->i', p - q, n)
        pc = p - p.mean(0)
        J = np.concatenate([np.cross(p - p.mean(0), n), n, np.einsum('ij,ij->i', pc, n)[:, None]], 1)
        hub = 0.02
        w = np.where(np.abs(r) < hub, 1.0, hub / np.abs(r))
        # damping proportional to the data: directions the overlap does not constrain
        # (e.g. sliding along the floor) stay put instead of drifting
        lam = len(r) * np.array([0.02] * 3 + [0.02] * 3 + [0.5])
        H = J.T @ (J * w[:, None]) + np.diag(lam)
        x = -np.linalg.solve(H, J.T @ (w * r))
        dR = _rotvec(x[:3])
        ds = 1 + x[6]
        pm = p.mean(0)
        # compose: new = ds * dR (P - pm) + pm + t   (rotation/scale about the matched centroid)
        R_tot = dR @ R_tot
        c_tot *= ds
        t_tot = ds * dR @ (t_tot + cen - pm) + pm + x[3:6] - cen
        if np.linalg.norm(x) < 1e-6:
            break
    P = c_tot * (B - cen) @ R_tot.T + cen + t_tot
    after = agreement(P, nB @ R_tot.T)
    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R_tot) - 1) / 2, -1, 1))))
    # express as X' = c R X + t in world coordinates
    t_world = cen + t_tot - c_tot * R_tot @ cen
    ok = (after > before + 0.02 and after > 1.15 * before and rot_deg <= max_rot_deg
          and np.linalg.norm(t_tot) <= max_trans and abs(c_tot - 1) <= max_scale_dev)
    info = {'applied': bool(ok), 'inliers_5cm_before': before, 'inliers_5cm_after': after,
            'rot_deg': rot_deg, 'trans': float(np.linalg.norm(t_tot)), 'scale': float(c_tot)}
    if ok:
        scene.apply_similarity(c_tot, R_tot, t_world, frames=scene.frames_of(AFTER))
    return info


def overlap_report(scene, voxel=0.05, radius=0.08):
    """Fraction of the after surface lying on the before surface"""
    A = session_points(scene, BEFORE, voxel).xyz
    B = session_points(scene, AFTER, voxel).xyz
    if len(A) == 0 or len(B) == 0:
        return 0.0
    d, _ = cKDTree(A).query(B)
    return float((d < radius).mean())
