"""
changes.py - object level reasoning: moved, added, removed or can't tell

Each object of one video is judged by the cameras of the other video:
before objects come out present, gone or unverified, after objects present,
new or unverified. Gone and new objects that look alike are matched as moved,
the rest are removed or added. Changed geometry that no detection explains
becomes an unidentified object.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .objects import Object3D
from .scene import AFTER, BEFORE, connected_components, dilate_keys, encode_voxels
from .visibility import CHANGED, FREE, SAME, DepthCache, evaluate


@dataclass
class Assessment:
    """Verdict for one object judged by the other video"""
    status: str               # present | gone | new | maybe | unverified
    frac_changed: float
    counts: dict
    confirming_views: list    # frames of the other recording that saw through the object
    looking_views: list       # frames of the other recording that looked at the location
    reason: str = ''


@dataclass
class Change:
    """One reported change"""
    id: str
    kind: str                 # moved | likely_moved | added | removed | appearance
    label: str
    confidence: float
    before: Object3D = None
    after: Object3D = None
    before_assess: Assessment = None
    after_assess: Assessment = None
    displacement: list = None
    yaw_deg: float = None
    sentence: str = ''
    details: dict = field(default_factory=dict)


@dataclass
class ChangeSet:
    """Everything the reasoning found"""
    changes: list
    unverified: list          # (Object3D, Assessment)
    unchanged: list           # (Object3D before, Object3D after | None)
    uncertain: list           # (Object3D, Assessment) partial evidence, not reported as change
    objects: dict             # session -> list[Object3D]
    assessments: dict         # object id -> Assessment
    stats: dict = field(default_factory=dict)


def assess_object(obj, scene, other_frames, cfg, cache):
    """Judges one object with the cameras of the other video"""
    ev = evaluate(obj.points, scene, other_frames, cfg, cache)
    c = ev.counts()
    n = len(obj.points)
    decisive = c['kept'] + c['changed']
    frac = c['changed'] / decisive if decisive else 0.0
    # per-view evidence: which other-recording frames looked at / saw through it
    looking, confirming = [], []
    for k, f in enumerate(ev.frames):
        s = int((ev.per_view[k] == SAME).sum())
        fr = int((ev.per_view[k] == FREE).sum())
        if s + fr >= max(10, 0.1 * n):
            looking.append(int(f))
            if fr >= max(8, 0.5 * (s + fr)):
                confirming.append(int(f))
    if decisive < max(15, cfg.min_evaluated_frac * n):
        reason = 'outside_view' if c['unseen'] >= c['hidden'] else 'occluded'
        return Assessment('unverified', frac, c, confirming, looking, reason)
    if frac >= cfg.object_gone_frac:
        status = 'gone' if obj.session == BEFORE else 'new'
    elif frac >= cfg.object_maybe_frac:
        status = 'maybe'
    else:
        status = 'present'
    return Assessment(status, frac, c, confirming, looking)


def _hs_hist(colors, bins=(12, 4)):
    """Hue-saturation histogram of point colours"""
    import cv2
    if len(colors) == 0:
        return np.ones(bins[0] * bins[1]) / (bins[0] * bins[1])
    hsv = cv2.cvtColor(colors.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h, _, _ = np.histogram2d(hsv[:, 0], hsv[:, 1], bins=bins, range=[[0, 180], [0, 256]])
    h = h.ravel() + 1e-6
    return h / h.sum()


def appearance_distance(a, b):
    """How different two objects look (CLIP, or colour histograms)"""
    if a.embedding is not None and b.embedding is not None:
        return float(1 - np.clip(a.embedding @ b.embedding, -1, 1))
    bc = float(np.sum(np.sqrt(_hs_hist(a.colors) * _hs_hist(b.colors))))   # bhattacharyya coeff
    return 1 - bc


def size_distance(a, b):
    """Mean log ratio of the sorted box sizes"""
    ea, eb = np.maximum(a.extents(), 0.03), np.maximum(b.extents(), 0.03)
    return float(np.mean(np.abs(np.log(ea / eb))))


def match_cost(a, b):
    """Cost of calling a and b the same object"""
    if a.source == 'region' or b.source == 'region':
        label_term = 0.25
    else:
        label_term = 0.0 if a.label == b.label else 0.7
    sz = size_distance(a, b)
    if min(len(a.points), len(b.points)) < 60:
        sz = 0.0                      # a few points of glass say nothing about size
    if sz > 0.75:                     # more than ~2x different in size: not the same thing
        return np.inf
    return label_term + appearance_distance(a, b) + 0.6 * sz


def estimate_motion(a, b):
    """Displacement between footprint centres, rotation from the long axis"""
    ca, cb = a.center, b.center
    disp = np.array([cb[0] - ca[0], cb[1] - ca[1], b.bottom - a.bottom])
    yaw = None
    _, _, wa, da, anga = a.footprint_box()
    _, _, wb, db, angb = b.footprint_box()
    if max(wa, da) / max(min(wa, da), 1e-3) >= 1.15 and max(wb, db) / max(min(wb, db), 1e-3) >= 1.15:
        ha = anga + (90 if da > wa else 0)          # direction of the long side
        hb = angb + (90 if db > wb else 0)
        yaw = float((hb - ha + 90) % 180 - 90)      # heading is defined modulo 180 deg
    return disp, yaw


def _rot2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def locate(template, cloud, guess, search=0.25, step=0.05, gate=0.04, max_points=500):
    """Pose (yaw, xy) that puts the template's footprint onto the cloud near guess, and the fit"""
    rng = np.random.default_rng(0)
    T = template if len(template) <= max_points else template[rng.choice(len(template), max_points, replace=False)]
    if len(cloud) < 20 or len(T) < 20:
        return None
    tree = cKDTree(cloud)
    tc = T[:, :2].mean(0)
    offs = np.arange(-search, search + 1e-9, step)
    best = (-1.0, 0.0, guess)
    # coarse: every 10 deg and 5 cm around the guess
    for th in np.deg2rad(np.arange(0, 360, 10)):
        base = (T[:, :2] - tc) @ _rot2(th).T
        for dx in offs:
            for dy in offs:
                t = guess + (dx, dy)
                d, _ = tree.query(np.c_[base + t, T[:, 2]], distance_upper_bound=gate)
                sc = np.isfinite(d).mean()
                if sc > best[0]:
                    best = (sc, th, t)
    _, th, t = best
    # fine: 2D point-to-point ICP with a shrinking gate
    for g in (0.06, 0.04, 0.03, 0.02):
        for _ in range(10):
            P = np.c_[(T[:, :2] - tc) @ _rot2(th).T + t, T[:, 2]]
            d, j = tree.query(P, distance_upper_bound=g)
            m = np.isfinite(d)
            if m.sum() < 10:
                break
            A, B = T[m, :2] - tc, cloud[j[m], :2]
            ma, mb = A.mean(0), B.mean(0)
            U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[1] *= -1
                R = Vt.T @ U.T
            th, t = np.arctan2(R[1, 0], R[0, 0]), mb - R @ ma
    P = np.c_[(T[:, :2] - tc) @ _rot2(th).T + t, T[:, 2]]
    d, _ = tree.query(P, distance_upper_bound=0.03)
    return float(th), t, tc, float(np.isfinite(d).mean())


def refine_motion(c, ps, cfg):
    """Measures a move by finding the better seen copy inside the other recording"""
    a, b = c.before, c.after
    big, small = (a, b) if len(a.points) >= len(b.points) else (b, a)
    cloud = ps[small.session].xyz
    lo, hi = big.bottom + 0.03, big.top + 0.05      # leave the floor out
    sc = small.center
    near = np.linalg.norm(cloud[:, :2] - sc[:2], axis=1) < 1.5
    cloud = cloud[near & (cloud[:, 2] > lo) & (cloud[:, 2] < hi)]
    r = locate(big.points[big.points[:, 2] > lo], cloud, sc[:2])
    if r is None:
        return
    th, t, tc, fit = r
    c.details['shape_fit'] = fit
    if fit < cfg.shape_min_fit:
        return
    # follow the footprint centre of the better seen copy, not the centroid of its seen
    # faces: under a turn, different points of an object move by different amounts
    p = big.center[:2]
    q = _rot2(th) @ (p - tc) + t
    if big is a:
        dxy, yaw = q - p, np.degrees(th)
    else:
        dxy, yaw = p - q, -np.degrees(th)
    yaw = (yaw + 180) % 360 - 180
    c.displacement = [float(dxy[0]), float(dxy[1]), c.displacement[2]]
    # a turn is only claimed when the shape fit and the footprint axis agree on it
    if c.yaw_deg is not None:
        diff = abs((yaw - c.yaw_deg + 90) % 180 - 90)
        c.yaw_deg = float(c.yaw_deg) if diff <= 20 else None
    c.details['measured_by'] = 'shape'


def find_regions(ps, ev, objects, cfg, prefix):
    """Changed geometry not explained by any detected object"""
    mask = (ev.verdict == CHANGED) & (ps.xyz[:, 2] > cfg.region_min_height)
    if objects:
        okeys = dilate_keys(np.unique(np.concatenate(
            [encode_voxels(o.points, cfg.object_voxel) for o in objects])))
        okeys = dilate_keys(okeys)
        mask &= ~np.isin(encode_voxels(ps.xyz, cfg.object_voxel), okeys)
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    lab = connected_components(ps.xyz[idx], cfg.region_voxel)
    regions = []
    for k in np.unique(lab):
        sel = idx[lab == k]
        if len(sel) < cfg.region_min_points:
            continue
        P = ps.xyz[sel]
        ext = np.percentile(P, 98, 0) - np.percentile(P, 2, 0)
        if ext.max() < cfg.region_min_extent:
            continue
        # thin sheets (a few cm thick, metres long) are misregistered walls, not objects
        sv = np.linalg.svd(P - P.mean(0), compute_uv=False) / np.sqrt(len(P))
        if sv[2] < 0.012 and ext.max() > 0.8:
            continue
        # without a name, a change has to look like a thing: long slivers are the
        # edges of doors, cabinets and lamps seen edge-on, where depth is least reliable
        e = np.sort(ext)
        if e[0] < cfg.region_min_thickness or e[2] > cfg.region_max_elongation * e[0]:
            continue
        if np.median(ev.n_free[sel]) < cfg.min_free_views:
            continue
        regions.append(Object3D(id=f'{prefix}{len(regions)}', session=ps.session,
                                label='unidentified object', score=0.0, points=P,
                                colors=ps.rgb[sel], frames=[], source='region'))
    return regions


def footprint_overlap(a, b, margin=0.0):
    """Overlap of the two footprints over the smaller one"""
    la, ha = np.percentile(a.points[:, :2], 5, 0) - margin, np.percentile(a.points[:, :2], 95, 0) + margin
    lb, hb = np.percentile(b.points[:, :2], 5, 0) - margin, np.percentile(b.points[:, :2], 95, 0) + margin
    inter = np.prod(np.clip(np.minimum(ha, hb) - np.maximum(la, lb), 0, None))
    small = min(np.prod(np.maximum(ha - la, 0.05)), np.prod(np.maximum(hb - lb, 0.05)))
    return float(inter / small)


def _overlap(a, b, voxel):
    """Fraction of the voxels of a inside b"""
    ka = np.unique(encode_voxels(a.points, voxel))
    kb = dilate_keys(np.unique(encode_voxels(b.points, voxel)))
    return float(np.isin(ka, kb).mean())


def detect_changes(scene, objs_before, objs_after, ps_before, ps_after,
                   ev_before, ev_after, cfg, dets=None):
    """Moved / added / removed / can't tell from the objects and the evidence"""
    cache = DepthCache(scene)
    fA, fB = scene.frames_of(BEFORE), scene.frames_of(AFTER)

    # unlabelled geometric changes become pseudo-objects
    regA = find_regions(ps_before, ev_before, objs_before, cfg, 'rA')
    regB = find_regions(ps_after, ev_after, objs_after, cfg, 'rB')

    assess = {}
    for o in objs_before + regA:
        assess[o.id] = assess_object(o, scene, fB, cfg, cache)
    for o in objs_after + regB:
        assess[o.id] = assess_object(o, scene, fA, cfg, cache)

    # an object whose own surface could not be checked (or gave weak / contradictory
    # evidence) while the other recording found the same kind of object on the same
    # footprint is simply being seen from another side: it is present. Typical case:
    # a shelf seen from the front in one video and from the back in the other
    # (unless its own evidence is overwhelming: then it really did move, just not far)
    def strong(a):
        return a.status in ('gone', 'new') and a.frac_changed >= 0.7 and len(a.confirming_views) >= 4
    snapshot = {k: v.status for k, v in assess.items()}
    for mine, theirs in ((objs_before, objs_after), (objs_after, objs_before)):
        for o in mine:
            a = assess[o.id]
            if a.status == 'present' or strong(a):
                continue
            same = [t for t in theirs if t.label == o.label and footprint_overlap(o, t, margin=0.2) >= 0.3]
            if not same:
                continue
            if a.status in ('unverified', 'maybe') or \
                    any(snapshot[t.id] in ('present', 'unverified', 'maybe') for t in same):
                a.status, a.reason = 'present', 'seen_from_other_side'

    # things depth sees through (glass, clear bottles): ask the detector instead
    if dets is not None and cfg.recheck:
        from .recheck import frames_by_index, not_redetected
        by_frame = frames_by_index(dets)
        for o in objs_before + objs_after:
            a = assess[o.id]
            if a.status not in ('present', 'unverified') or a.reason == 'seen_from_other_side' \
                    or o.extents()[1] > cfg.recheck_max_size:
                continue
            looks = not_redetected(o, scene, by_frame, cfg, cache)
            if looks:
                a.status = 'gone' if o.session == BEFORE else 'new'
                a.reason, a.confirming_views, a.looking_views = 'not_redetected', looks, looks

    # regions must themselves be clearly changed, and not just the un-masked part
    # (legs, edges) of a detected object that changed
    changed_objs = [o for o in objs_before + objs_after if assess[o.id].status in ('gone', 'new', 'maybe')]
    def absorbed(r):
        for o in changed_objs:
            if o.session != r.session:
                continue
            d, _ = cKDTree(o.points).query(r.points)
            if np.median(d) < 0.25:
                return True
        return False
    regA = [r for r in regA if assess[r.id].status == 'gone' and not absorbed(r)]
    regB = [r for r in regB if assess[r.id].status == 'new' and not absorbed(r)]

    candA = [o for o in objs_before + regA if assess[o.id].status in ('gone', 'maybe')]
    candB = [o for o in objs_after + regB if assess[o.id].status in ('new', 'maybe')]

    changes, used_a, used_b = [], set(), set()

    # 1. moved: optimal assignment between gone and new
    if candA and candB:
        C = np.array([[match_cost(a, b) for b in candB] for a in candA])
        # every object may also stay unpaired at the cost of the match threshold, so one
        # impossible pair can't force the others into worse ones
        nA, nB = C.shape
        big = np.full((nA + nB, nB + nA), 1e6)
        big[:nA, :nB] = np.where(np.isfinite(C), C, 1e6)
        big[:nA, nB:][np.eye(nA, dtype=bool)] = 0.9
        big[nA:, :nB][np.eye(nB, dtype=bool)] = 0.9
        big[nA:, nB:] = 0.0
        ri, ci = linear_sum_assignment(big)
        for i, j in zip(ri, ci):
            if i >= nA or j >= nB or C[i, j] > 0.9:
                continue
            a, b = candA[i], candB[j]
            if assess[a.id].status == 'maybe' and assess[b.id].status == 'maybe':
                continue
            disp, yaw = estimate_motion(a, b)
            if np.linalg.norm(disp[:2]) < cfg.moved_min_dist and abs(yaw or 0) < cfg.moved_min_yaw:
                used_a.add(a.id), used_b.add(b.id)        # same place: noise, not a move
                continue
            label = a.label if a.source == 'detected' else b.label
            conf = 0.5 * (_conf(assess[a.id]) + _conf(assess[b.id])) * (1 - 0.3 * min(C[i, j], 1))
            changes.append(Change('', 'moved', label, conf, a, b, assess[a.id], assess[b.id],
                                  displacement=disp.tolist(), yaw_deg=yaw,
                                  details={'match_cost': float(C[i, j])}))
            used_a.add(a.id), used_b.add(b.id)

    # 1b. clearly new (or gone) right where the same kind of object still stands in the
    # other recording: it was shifted or turned in place (a chair pushed round a table)
    moved_in_place = set()
    for o in candA + candB:
        if o.source != 'detected' or o.id in used_a or o.id in used_b or not strong(assess[o.id]):
            continue
        theirs = objs_after if o.session == BEFORE else objs_before
        same = [t for t in theirs if t.label == o.label and assess[t.id].status == 'present'
                and t.id not in moved_in_place and footprint_overlap(o, t, margin=0.2) >= 0.3]
        if not same:
            continue
        t = max(same, key=lambda t: footprint_overlap(o, t, margin=0.2))
        a, b = (o, t) if o.session == BEFORE else (t, o)
        disp, yaw = estimate_motion(a, b)
        conf = 0.8 * _conf(assess[o.id])
        changes.append(Change('', 'moved', o.label, conf, a, b, assess[a.id], assess[b.id],
                              displacement=disp.tolist(), yaw_deg=yaw, details={'in_place': True}))
        (used_a if o.session == BEFORE else used_b).add(o.id)
        moved_in_place.add(t.id)

    # 2. gone/new whose counterpart is somewhere the other recording never saw
    unver_after = [o for o in objs_after if assess[o.id].status == 'unverified']
    unver_before = [o for o in objs_before if assess[o.id].status == 'unverified']
    for a in [o for o in candA if o.id not in used_a and assess[o.id].status == 'gone']:
        best = _best_lookalike(a, unver_after, strict=True)
        if best is not None:
            disp, yaw = estimate_motion(a, best)
            sure = 1.0 if assess[a.id].reason == 'not_redetected' else 0.5
            changes.append(Change('', 'likely_moved', a.label, sure * _conf(assess[a.id]), a, best,
                                  assess[a.id], assess[best.id], disp.tolist(), yaw))
            used_a.add(a.id)
            unver_after.remove(best)
    for b in [o for o in candB if o.id not in used_b and assess[o.id].status == 'new']:
        best = _best_lookalike(b, unver_before, strict=True)
        if best is not None:
            disp, yaw = estimate_motion(best, b)
            changes.append(Change('', 'likely_moved', b.label, 0.5 * _conf(assess[b.id]), best, b,
                                  assess[best.id], assess[b.id], disp.tolist(), yaw))
            used_b.add(b.id)
            unver_before.remove(best)

    # 3. removed / added
    uncertain = []
    for a in candA:
        if a.id in used_a:
            continue
        if assess[a.id].status == 'gone':
            changes.append(Change('', 'removed', a.label, _conf(assess[a.id]) * (0.8 if a.source == 'region' else 1),
                                  before=a, before_assess=assess[a.id]))
        else:
            uncertain.append((a, assess[a.id]))
    for b in candB:
        if b.id in used_b:
            continue
        if assess[b.id].status == 'new':
            changes.append(Change('', 'added', b.label, _conf(assess[b.id]) * (0.8 if b.source == 'region' else 1),
                                  after=b, after_assess=assess[b.id]))
        else:
            uncertain.append((b, assess[b.id]))

    # 4. unchanged objects (also the landmarks used in descriptions)
    unchanged = []
    presentB = [o for o in objs_after if assess[o.id].status == 'present' and o.id not in moved_in_place]
    paired = set()
    for a in [o for o in objs_before if assess[o.id].status == 'present' and o.id not in moved_in_place]:
        best, bo = None, 0.2
        for b in presentB:
            ov = max(_overlap(a, b, cfg.object_voxel), _overlap(b, a, cfg.object_voxel),
                     footprint_overlap(a, b, margin=0.2) if a.label == b.label else 0.0)
            if ov > bo:
                best, bo = b, ov
        if best is not None and best.id in paired:
            continue                      # a second piece of an already listed object
        if best is not None:
            paired.add(best.id)
        unchanged.append((a, best))
    for b in presentB:
        if b.id not in paired:
            unchanged.append((None, b))

    # a small unchecked piece right next to a checked object of the same kind is part of it
    # (a desk's leg seen on its own), not a separate thing to worry about
    def kind(o):
        return 'table' if o.label in ('table', 'desk') else o.label
    present = [o for o in objs_before + objs_after if assess[o.id].status == 'present']
    def piece(o):
        return any(kind(p) == kind(o) and len(p.points) > 2 * len(o.points)
                   and footprint_overlap(o, p, margin=0.3) >= 0.3 for p in present)
    unverified = [(o, assess[o.id]) for o in unver_before + unver_after if not piece(o)]

    for c in changes:
        seen = all(x is None or x.reason != 'not_redetected' for x in (c.before_assess, c.after_assess))
        if c.kind in ('moved', 'likely_moved') and c.before.source == c.after.source == 'detected' and seen:
            refine_motion(c, {BEFORE: ps_before, AFTER: ps_after}, cfg)

    order = {'moved': 0, 'likely_moved': 1, 'added': 2, 'removed': 3}
    changes.sort(key=lambda c: (order[c.kind], -c.confidence))
    for k, c in enumerate(changes):
        c.id = f'C{k + 1}'

    return ChangeSet(changes, unverified, unchanged, uncertain,
                     {BEFORE: objs_before + regA, AFTER: objs_after + regB}, assess,
                     stats={'regions_before': len(regA), 'regions_after': len(regB)})


def _best_lookalike(o, pool, strict=True):
    """Most similar object in pool, or None"""
    best, bc = None, 0.45 if strict else 0.9
    for p in pool:
        if o.source == 'detected' and p.label != o.label:
            continue
        c = match_cost(o, p)
        if c < bc:
            best, bc = p, c
    return best


def _conf(a):
    """Confidence from the see-through fraction and the number of views"""
    views = len(a.confirming_views)
    if a.reason == 'not_redetected':  # the detector's word only
        return float(np.clip(0.35 + 0.05 * views, 0, 0.6))
    return float(np.clip(0.55 * a.frac_changed + 0.45 * min(1.0, views / 4), 0, 1))
