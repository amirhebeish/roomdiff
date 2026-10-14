"""
expect.py - renders what a camera should see if nothing had changed

The first video is used as the memory of the room. Its points are splatted
with a z-buffer into a camera of the second video, which gives the expected
image and depth from exactly that viewpoint. Pixels the memory never covered
stay empty: never seen before.
"""

import numpy as np
from scipy import ndimage


class MemoryRenderer:
    """Renders one video from the cameras of the other"""
    def __init__(self, scene, session, stride=1, max_points=4_000_000):
        pts, cols = [], []
        for i in scene.frames_of(session):
            p, c, _ = scene.backproject(i, stride=stride)
            pts.append(p)
            cols.append(c)
        P, C = np.concatenate(pts), np.concatenate(cols)
        if len(P) > max_points:
            sel = np.random.default_rng(0).choice(len(P), max_points, replace=False)
            P, C = P[sel], C[sel]
        self.scene, self.session, self.P, self.C = scene, session, P, C

    def render(self, frame, splat=2, fill_iters=2):
        """Expected rgb, depth (NaN = never seen) and the memory point behind each pixel"""
        s = self.scene
        H, W = s.hw
        u, v, z = s.project(self.P, frame)
        ok = (z > 0.05) & (u > -1) & (u < W) & (v > -1) & (v < H)
        u, v, z, col = u[ok], v[ok], z[ok], self.C[ok]
        zbuf = np.full(H * W, np.inf, np.float32)
        lins, zs, idxs = [], [], []
        for du in range(splat):
            for dv in range(splat):
                ui = np.floor(u).astype(int) + du
                vi = np.floor(v).astype(int) + dv
                m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                lins.append(vi[m] * W + ui[m])
                zs.append(z[m])
                idxs.append(np.nonzero(m)[0])
        lin, zz, idx = np.concatenate(lins), np.concatenate(zs), np.concatenate(idxs)
        np.minimum.at(zbuf, lin, zz)
        win = zz <= zbuf[lin] * 1.0005
        rgb = np.zeros((H * W, 3), np.uint8)
        rgb[lin[win]] = col[idx[win]]
        src = np.full(H * W, -1, np.int64)                 # which memory point won each pixel
        src[lin[win]] = np.nonzero(ok)[0][idx[win]]
        depth = np.where(np.isfinite(zbuf), zbuf, np.nan).reshape(H, W)
        rgb = rgb.reshape(H, W, 3)
        src = src.reshape(H, W)
        # fill pinholes: a missing pixel surrounded by memory takes its nearest neighbour
        for _ in range(fill_iters):
            miss = ~np.isfinite(depth)
            if not miss.any():
                break
            known = ~miss
            n = ndimage.uniform_filter(known.astype(np.float32), 3) * 9
            cand = miss & (n >= 5)
            if not cand.any():
                break
            dfill = ndimage.minimum_filter(np.where(known, depth, np.inf), 3)
            _, (iy, ix) = ndimage.distance_transform_edt(miss, return_indices=True)
            depth[cand] = dfill[cand]
            rgb[cand] = rgb[iy[cand], ix[cand]]
            src[cand] = src[iy[cand], ix[cand]]
        return rgb, depth, src


def verdict_lookup(ps, ev, voxel):
    """Function giving the visibility verdict of the voxel each point falls in"""
    from .scene import encode_voxels
    keys = encode_voxels(ps.xyz, voxel)
    order = np.argsort(keys)
    sk, sv = keys[order], ev.verdict[order]

    def lookup(xyz):
        k = encode_voxels(xyz, voxel)
        j = np.clip(np.searchsorted(sk, k), 0, len(sk) - 1)
        out = np.where(sk[j] == k, sv[j], 0).astype(np.uint8)
        return out
    return lookup
