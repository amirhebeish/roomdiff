"""
objects.py - lifts 2D masks into 3D and fuses them into objects
"""

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CANONICAL, MAX_SIZE, MIN_ASPECT, MIN_SIZE, SURE_NAME_VOTES
from .scene import connected_components, dilate_keys, encode_voxels, voxel_downsample


@dataclass
class Detection:
    """One 2D detection with its mask"""
    frame: int                     # index into Scene
    label: str
    score: float
    box: tuple                     # xyxy in depth-resolution pixels
    mask: np.ndarray               # (H, W) bool at depth resolution
    embedding: np.ndarray = None   # appearance descriptor (CLIP) if available


@dataclass
class Object3D:
    """One object in 3D (room frame: z up, floor at z = 0)"""
    id: str
    session: int
    label: str
    score: float
    points: np.ndarray
    colors: np.ndarray
    frames: list[int]
    label_votes: dict = field(default_factory=dict)
    embedding: np.ndarray = None
    source: str = 'detected'       # "detected" (has a semantic label) or "region" (geometry only)

    # geometry helpers (room frame: z up, floor at z = 0)
    @property
    def sure(self):
        """Whether the name is backed by enough confident detections to be shown"""
        if self.source != 'detected':
            return True
        if max(self.label_votes.values(), default=0.0) < SURE_NAME_VOTES:
            return False
        side = max(self.extents()[:2])
        return MIN_SIZE.get(self.label, 0.0) <= side <= 1.3 * MAX_SIZE.get(self.label, np.inf)

    @property
    def centroid(self):
        return np.median(self.points, 0)

    @property
    def center(self):
        """Centre of the footprint box, robust to seeing only one side"""
        cx, cy, _, _, _ = self.footprint_box()
        return np.array([cx, cy, 0.5 * (self.bottom + self.top)])

    @property
    def bottom(self):
        return float(np.percentile(self.points[:, 2], 3))

    @property
    def top(self):
        return float(np.percentile(self.points[:, 2], 97))

    @property
    def surface(self):
        """Height of its main horizontal surface (a table top, not the lamp on it)"""
        z = self.points[:, 2]
        h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.04, 0.02))
        k = int(np.argmax(h))
        return float(max(e[k + 1], self.top - 0.3)) if h.size else self.top

    def footprint_box(self):
        """Minimum area rectangle of the footprint: cx, cy, w, d, angle"""
        xy = self.points[:, :2].astype(np.float32)
        (cx, cy), (w, d), ang = cv2.minAreaRect(xy)
        return float(cx), float(cy), float(w), float(d), float(ang)

    def box(self):
        """3D box: centre, size, yaw"""
        cx, cy, w, d, ang = self.footprint_box()
        lo, hi = self.bottom, self.top
        return {'center': [cx, cy, (lo + hi) / 2], 'size': [w, d, max(hi - lo, 0.02)], 'yaw': ang}

    def extents(self):
        """Sorted footprint sides and height"""
        _, _, w, d, _ = self.footprint_box()
        return np.array(sorted([w, d]) + [self.top - self.bottom])

    def size_text(self):
        _, _, w, d, _ = self.footprint_box()
        w, d = max(w, d), min(w, d)
        h = self.top - self.bottom
        return f'{w*100:.0f} x {d*100:.0f} x {h*100:.0f} cm'


def main_copy(a, b):
    """The more complete of the two copies of one object (either may be None)"""
    return max([x for x in (a, b) if x is not None], key=lambda x: len(x.points))


def pair_sure(a, b):
    """Whether an unchanged object's name is shown as fact, judged on its more complete copy"""
    return main_copy(a, b).sure


def canonical(label, vocab):
    """Maps a detector phrase onto the vocabulary, then onto one name"""
    label = label.strip().lower()
    if label not in vocab:
        hits = [v for v in vocab if v in label]
        if hits:
            label = max(hits, key=len)
        else:
            words = label.split()
            hits = [v for v in vocab if words and v.split()[-1] == words[-1]]
            # Grounding DINO sometimes returns part of a phrase ("trash" for "trash can")
            part = [v for v in vocab if words and any(w in v.split() for w in words)]
            label = hits[0] if hits else (part[0] if part else label)
    return CANONICAL.get(label, label)


def lift_detection(scene, det, voxel, min_points, max_extent):
    """World points of a detection, largest 3D component only"""
    m = det.mask.astype(np.uint8)
    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    pts, cols, _ = scene.backproject(det.frame, mask=m)
    if len(pts) < min_points:
        return None
    lab = connected_components(pts, voxel)
    best = np.bincount(lab).argmax()
    keep = lab == best
    pts, cols = pts[keep], cols[keep]
    if len(pts) < min_points:
        return None
    ext = np.percentile(pts, 98, 0) - np.percentile(pts, 2, 0)
    if ext.max() > max_extent:
        return None
    return pts, cols


def fuse_objects(scene, dets, session, cfg, prefix):
    """Greedy fusion of the lifted detections of one video into objects"""
    vox = cfg.object_voxel
    lifted = []
    for d in sorted(dets, key=lambda d: -d.score):
        if scene.session[d.frame] != session:
            continue
        r = lift_detection(scene, d, vox, cfg.min_object_points, cfg.max_object_extent)
        if r is None:
            continue
        keys = np.unique(encode_voxels(r[0], vox))
        lifted.append((d, r[0], r[1], keys))

    def aabb(p):
        return np.percentile(p, 5, 0), np.percentile(p, 95, 0)

    def box_overlap(a, b):
        lo, hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
        inter = np.prod(np.clip(hi - lo, 0, None))
        va = np.prod(np.maximum(a[1] - a[0], 0.02))
        vb = np.prod(np.maximum(b[1] - b[0], 0.02))
        return inter / min(va, vb), inter / (va + vb - inter)

    clusters = []   # each: dict(keys, dil, members, label, box)
    for d, pts, cols, keys in lifted:
        best, best_score = None, 0.0
        bx = aabb(pts)
        for c in clusters:
            inter_det = np.isin(keys, c['dil'], assume_unique=True).sum() / len(keys)
            inter_obj = np.isin(c['keys'], dilate_keys(keys), assume_unique=True).sum() / len(c['keys'])
            contain = max(inter_det, inter_obj)
            bcont, biou = box_overlap(bx, c['box'])
            same_label = d.label == c['label']
            # different views see different faces of an object, so surfaces may barely
            # overlap while their bounding boxes do: use both cues
            if same_label:
                ok = contain >= 0.25 or bcont >= 0.5
            else:
                ok = contain >= 0.5 or biou >= 0.5
            sc = contain + bcont + (0.5 if same_label else 0)
            if ok and sc > best_score:
                best, best_score = c, sc
        if best is None:
            clusters.append({'keys': keys, 'dil': dilate_keys(keys), 'members': [(d, pts, cols)],
                             'label': d.label, 'box': bx})
        else:
            best['members'].append((d, pts, cols))
            best['keys'] = np.union1d(best['keys'], keys)
            best['dil'] = dilate_keys(best['keys'])
            best['box'] = aabb(np.concatenate([m[1] for m in best['members']]))

    objects = []
    for c in clusters:
        frames = sorted({m[0].frame for m in c['members']})
        if len(frames) < cfg.min_views_per_object:
            continue
        votes = defaultdict(float)
        for m in c['members']:
            votes[m[0].label] += m[0].score
        pts = np.concatenate([m[1] for m in c['members']])
        cols = np.concatenate([m[2] for m in c['members']])
        # keep only voxels supported by >= 2 views when enough views exist (removes mask bleed)
        if len(frames) >= 3:
            per_view_keys = [np.unique(encode_voxels(m[1], vox)) for m in c['members']]
            allk = np.concatenate(per_view_keys)
            uk, cnt = np.unique(allk, return_counts=True)
            good = uk[cnt >= 2]
            keep = np.isin(encode_voxels(pts, vox), good)
            if keep.sum() >= cfg.min_object_points and keep.mean() >= 0.5:
                pts, cols = pts[keep], cols[keep]
        lab = connected_components(pts, vox)
        keep = lab == np.bincount(lab).argmax()
        pts, cols = pts[keep], cols[keep]
        pts, cols = voxel_downsample(pts, cols, cfg.voxel)
        label = plausible_label(votes, pts)
        embs = [m[0].embedding for m in c['members'] if m[0].embedding is not None]
        emb = None
        if embs:
            emb = np.mean(embs, 0)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
        objects.append(Object3D(id='', session=session, label=label,
                                score=float(np.mean([m[0].score for m in c['members']])),
                                points=pts, colors=cols, frames=frames,
                                label_votes=dict(votes), embedding=emb))
    objects = _merge_duplicates(objects, cfg)
    mark_desks(objects)
    objects.sort(key=lambda o: (o.label, -len(o.points)))
    for k, o in enumerate(objects):
        o.id = f'{prefix}{k}'
    return objects


def plausible_label(votes, pts):
    """Most voted name that fits the measured size (close votes only)"""
    ranked = sorted(votes, key=votes.get, reverse=True)
    xy = pts[:, :2].astype(np.float32)
    sides = sorted(cv2.minAreaRect(xy)[1]) if len(xy) >= 3 else [0.0, 0.0]
    aspect = sides[1] / max(sides[0], 1e-3)
    for lab in ranked:
        if votes[lab] < 0.5 * votes[ranked[0]]:
            break
        if sides[1] <= 1.2 * MAX_SIZE.get(lab, np.inf) and aspect >= MIN_ASPECT.get(lab, 0.0):
            return lab
    return ranked[0]


def mark_desks(objects):
    """A table with a monitor on it is called a desk, a screen too wide for a desk a TV"""
    for o in objects:
        if o.label == 'monitor' and max(o.extents()[:2]) > 0.7:
            o.label = 'TV'
    tops = [o for o in objects if o.label == 'monitor']
    for o in objects:
        if o.label != 'table':
            continue
        lo, hi = np.percentile(o.points[:, :2], 3, 0), np.percentile(o.points[:, :2], 97, 0)
        for t in tops:
            c = t.center
            if np.all(c[:2] >= lo - 0.1) and np.all(c[:2] <= hi + 0.1) and abs(t.bottom - o.surface) < 0.2:
                o.label = 'desk'
                break


def _merge_duplicates(objects, cfg):
    """Merges same label objects on the same footprint"""
    from .changes import footprint_overlap
    merged = True
    while merged:
        merged = False
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                if a.label != b.label:
                    continue
                z_overlap = min(a.top, b.top) - max(a.bottom, b.bottom)
                if footprint_overlap(a, b, margin=0.1) >= 0.4 and z_overlap > 0:
                    pts = np.concatenate([a.points, b.points])
                    cols = np.concatenate([a.colors, b.colors])
                    pts, cols = voxel_downsample(pts, cols, cfg.voxel)
                    votes = dict(a.label_votes)
                    for k, v in b.label_votes.items():
                        votes[k] = votes.get(k, 0) + v
                    emb = a.embedding
                    if a.embedding is not None and b.embedding is not None:
                        emb = a.embedding + b.embedding
                        emb = emb / (np.linalg.norm(emb) + 1e-9)
                    objects[i] = Object3D(a.id, a.session, a.label, max(a.score, b.score), pts, cols,
                                          sorted(set(a.frames) | set(b.frames)), votes, emb)
                    del objects[j]
                    merged = True
                    break
            if merged:
                break
    return objects


def label_from_detections(scene, obj, dets, cache=None, min_frac=0.5):
    """Label for an unlabelled 3D region from the 2D detections that cover it, or None"""
    from .visibility import SAME, ray_check
    frames = scene.frames_of(obj.session)
    by_frame = {}
    for d in dets:
        by_frame.setdefault(d.frame, []).append(d)
    pts = obj.points
    if len(pts) > 3000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
    pv = ray_check(pts, scene, frames, cache=cache)
    H, W = scene.hw
    votes, n_frames = defaultdict(float), defaultdict(int)
    for k, f in enumerate(frames):
        vis = pts[pv[k] == SAME]
        if len(vis) < 20 or f not in by_frame:
            continue
        u, v, _ = scene.project(vis, f)
        ui, vi = np.clip(np.round(u).astype(int), 0, W - 1), np.clip(np.round(v).astype(int), 0, H - 1)
        for d in by_frame[f]:
            # fraction of the visible region inside this detection's mask
            frac = d.mask[vi, ui].mean()
            if frac >= min_frac:
                votes[d.label] += d.score * frac
                n_frames[d.label] += 1
    if not votes:
        return None
    best = max(votes, key=votes.get)
    # needs two views and a clear majority
    if n_frames[best] >= 2 and votes[best] >= 0.6 * sum(votes.values()):
        return best
    return None
