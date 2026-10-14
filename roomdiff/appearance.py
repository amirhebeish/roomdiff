"""
appearance.py - changes that are visible but have no height (rugs, mats, papers)

On surfaces the depth test confirmed unchanged, the two videos should also
agree in colour. One global colour map takes out the exposure and white balance
difference between the videos, then the colours are compared voxel by voxel
with chroma weighted over brightness (shadows change brightness, objects change
colour). Comparing each region with its surroundings in each video tells
removed from added.
"""

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .objects import Object3D
from .scene import AFTER, BEFORE, connected_components, dilate_keys, encode_voxels
from .visibility import KEPT


def _lab(rgb):
    """RGB to Lab"""
    return cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)


def _voxel_means(xyz, lab, voxel):
    """Mean colour per voxel"""
    keys = encode_voxels(xyz, voxel)
    uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    mean = np.stack([np.bincount(inv, weights=lab[:, k], minlength=len(uk)) for k in range(3)], 1) / cnt[:, None]
    return uk, mean, cnt


def _robust_affine(src, dst, iters=5):
    """Affine colour map with outliers down weighted"""
    X = np.concatenate([src, np.ones((len(src), 1))], 1)
    w = np.ones(len(src))
    M = np.zeros((4, 3))
    for _ in range(iters):
        Xw = X * w[:, None]
        M = np.linalg.lstsq(Xw.T @ X + 1e-6 * np.eye(4), Xw.T @ dst, rcond=None)[0]
        r = np.linalg.norm(X @ M - dst, axis=1)
        c = 2.0 * np.median(r) + 1e-6
        w = np.where(r < c, 1.0, c / r)
    return M


def find_appearance_changes(ps, ev, changed_objects, cfg):
    """Colour-only changes as (region before, region after, info)"""
    vox = cfg.app_voxel
    sel = {s: ev[s].verdict == KEPT for s in (BEFORE, AFTER)}
    kA, mA, cA = _voxel_means(ps[BEFORE].xyz[sel[BEFORE]], _lab(ps[BEFORE].rgb[sel[BEFORE]]), vox)
    kB, mB, cB = _voxel_means(ps[AFTER].xyz[sel[AFTER]], _lab(ps[AFTER].rgb[sel[AFTER]]), vox)
    common, ia, ib = np.intersect1d(kA, kB, return_indices=True)
    good = (cA[ia] >= 3) & (cB[ib] >= 3)
    common, ia, ib = common[good], ia[good], ib[good]
    if len(common) < 200:
        return [], {'pairs': int(len(common))}
    M = _robust_affine(mB[ib], mA[ia])
    corrB = np.concatenate([mB[ib], np.ones((len(ib), 1))], 1) @ M
    d = mA[ia] - corrB
    dist = np.sqrt((cfg.app_luma_weight * d[:, 0]) ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)
    flagged = dist > cfg.app_thresh

    # misalignment guard: at a colour boundary a 2-3 cm registration error makes
    # colours disagree. A voxel only counts if NO neighbouring voxel of the other
    # video explains its colour (same idea as the 3x3 window in the depth test)
    if flagged.any():
        def dmetric(x, y):
            e = x - y
            return np.sqrt((cfg.app_luma_weight * e[..., 0]) ** 2 + e[..., 1] ** 2 + e[..., 2] ** 2)
        corr_all_B = np.concatenate([mB, np.ones((len(mB), 1))], 1) @ M
        fi = np.nonzero(flagged)[0]
        nbB = _neighbour_lookup(kB, common[fi])          # (F, 27) indices into kB or -1
        nbA = _neighbour_lookup(kA, common[fi])
        best_ab = np.where(nbB >= 0, dmetric(mA[ia[fi]][:, None, :], corr_all_B[np.maximum(nbB, 0)]), np.inf).min(1)
        best_ba = np.where(nbA >= 0, dmetric(mA[np.maximum(nbA, 0)], corrB[fi][:, None, :]), np.inf).min(1)
        flagged[fi[(best_ab <= cfg.app_thresh) | (best_ba <= cfg.app_thresh)]] = False

    # guard: nothing within reach of a geometric change (shadows, occlusion edges)
    centers = (np.stack([(common >> 42) & 0x1FFFFF, (common >> 21) & 0x1FFFFF, common & 0x1FFFFF], 1)
               - (1 << 20) + 0.5) * vox
    if changed_objects:
        P = np.concatenate([o.points for o in changed_objects])
        dd, _ = cKDTree(P).query(centers, distance_upper_bound=cfg.app_shadow_guard)
        flagged &= ~np.isfinite(dd)
    if flagged.sum() < cfg.app_min_voxels:
        return [], {'pairs': int(len(common)), 'flagged': int(flagged.sum())}

    lab = connected_components(centers[flagged], vox)
    fkeys, fdist = common[flagged], dist[flagged]
    out = []
    for k in np.unique(lab):
        m = lab == k
        if m.sum() < cfg.app_min_voxels:
            continue
        keys = fkeys[m]
        c = centers[flagged][m]
        if np.ptp(c[:, 0]) < 0.1 and np.ptp(c[:, 1]) < 0.1 or \
                sorted(np.ptp(c, 0))[1] < cfg.app_min_width - vox:
            continue                                   # slivers are boundary artefacts, not objects
        ring = np.setdiff1d(dilate_keys(dilate_keys(keys)), keys)
        regions = {}
        for s, (kk, mm) in ((BEFORE, (kA, mA)), (AFTER, (kB, mB))):
            in_reg = np.isin(encode_voxels(ps[s].xyz, vox), keys) & sel[s]
            regions[s] = Object3D(id='', session=s, label='flat item', score=0.0, points=ps[s].xyz[in_reg],
                                  colors=ps[s].rgb[in_reg], frames=[], source='appearance')
            jr = np.isin(kk, ring)
            jg = np.isin(kk, keys)
            if jr.any() and jg.any():
                regions[s].ring_contrast = float(np.linalg.norm(mm[jg].mean(0) - mm[jr].mean(0)))
            else:
                regions[s].ring_contrast = 0.0
        a, b = regions[BEFORE], regions[AFTER]
        if len(a.points) < 20 or len(b.points) < 20:
            continue
        # direction: which video's region stands out from its own surroundings?
        if a.ring_contrast > 1.8 * b.ring_contrast:
            direction = 'removed'
        elif b.ring_contrast > 1.8 * a.ring_contrast:
            direction = 'added'
        else:
            direction = 'changed'
        out.append((a, b, {'direction': direction, 'colour_difference': float(np.median(fdist[m])),
                           'area_m2': float(m.sum() * vox * vox)}))
    return out, {'pairs': int(len(common)), 'flagged': int(flagged.sum()), 'colour_map': M.tolist()}


_OFFS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], np.int64)


def _neighbour_lookup(sorted_keys, keys):
    """Indices of the 27 neighbouring voxels of each key, -1 if missing"""
    offs = (_OFFS[:, 0] << 42) + (_OFFS[:, 1] << 21) + _OFFS[:, 2]
    q = keys[:, None] + offs[None, :]
    j = np.clip(np.searchsorted(sorted_keys, q), 0, len(sorted_keys) - 1)
    return np.where(sorted_keys[j] == q, j, -1)
