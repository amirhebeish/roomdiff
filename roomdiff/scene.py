"""
scene.py - the shared scene used by every stage

Holds the frames of both videos in one world frame: images, depth, intrinsics,
world to camera poses (OpenCV convention: x right, y down, z forward) and which
video each frame came from.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BEFORE, AFTER = 0, 1
SESSION_NAMES = {BEFORE: 'before', AFTER: 'after'}


@dataclass
class Scene:
    """Frames of both videos in one world frame"""
    images: np.ndarray          # (N, H, W, 3) uint8, same resolution as depth
    depth: np.ndarray           # (N, H, W) float32, NaN where invalid
    K: np.ndarray               # (N, 3, 3) pinhole intrinsics in pixels
    T_cw: np.ndarray            # (N, 4, 4) world -> camera
    session: np.ndarray         # (N,) int, 0 = before, 1 = after
    frame_names: list[str] = field(default_factory=list)
    hires_paths: list[str] = None   # optional higher-resolution frames (same aspect, no crop)
    units: str = 'relative'     # becomes "m" once metric scale is estimated
    meta: dict = field(default_factory=dict)

    # basics
    @property
    def n(self):
        return int(self.depth.shape[0])

    @property
    def hw(self):
        return int(self.depth.shape[1]), int(self.depth.shape[2])

    def frames_of(self, s):
        """Frame indices of one video"""
        return np.nonzero(self.session == s)[0]

    def cam_centers(self, idx=None):
        """Camera centres in the world frame"""
        T = self.T_cw if idx is None else self.T_cw[idx]
        R, t = T[..., :3, :3], T[..., :3, 3]
        return -np.einsum('...ji,...j->...i', R, t)

    def cam_axes_world(self, i):
        """Camera right, down and forward axes in the world frame"""
        R = self.T_cw[i, :3, :3]
        return {'right': R[0], 'down': R[1], 'forward': R[2]}

    # (un)projection
    def backproject(self, i, mask=None, stride=1):
        """World points, colours and pixels of the valid depth of frame i"""
        D = self.depth[i]
        valid = np.isfinite(D) & (D > 0)
        if mask is not None:
            valid &= mask
        if stride > 1:
            sub = np.zeros_like(valid)
            sub[::stride, ::stride] = True
            valid &= sub
        v, u = np.nonzero(valid)
        z = D[v, u]
        K = self.K[i]
        x = (u - K[0, 2]) / K[0, 0] * z
        y = (v - K[1, 2]) / K[1, 1] * z
        Pc = np.stack([x, y, z], 1)
        R, t = self.T_cw[i, :3, :3], self.T_cw[i, :3, 3]
        Pw = (Pc - t) @ R          # R^T (Pc - t)
        return Pw.astype(np.float32), self.images[i][v, u], np.stack([v, u], 1)

    def project(self, points, i):
        """Pixel u, v and depth z of world points in frame i"""
        R, t = self.T_cw[i, :3, :3], self.T_cw[i, :3, 3]
        Pc = points @ R.T + t
        z = Pc[:, 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        K = self.K[i]
        u = K[0, 0] * Pc[:, 0] / zs + K[0, 2]
        v = K[1, 1] * Pc[:, 1] / zs + K[1, 2]
        return u, v, z

    # world transforms
    def apply_similarity(self, c, R, t, frames=None):
        """Moves (some of) the scene by X' = c R X + t"""
        idx = np.arange(self.n) if frames is None else np.asarray(frames)
        for i in idx:
            Rc, tc = self.T_cw[i, :3, :3].copy(), self.T_cw[i, :3, 3].copy()
            Rn = Rc @ R.T
            tn = -Rn @ t + c * tc
            self.T_cw[i, :3, :3] = Rn
            self.T_cw[i, :3, 3] = tn
            self.depth[i] = self.depth[i] * c

    # persistence
    def save(self, path):
        path = Path(path)
        np.savez_compressed(
            path, images=self.images, depth=self.depth.astype(np.float32), K=self.K,
            T_cw=self.T_cw, session=self.session,
            frame_names=np.array(self.frame_names),
            hires_paths=np.array(self.hires_paths or []),
            units=np.array(self.units), meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        hires = [str(x) for x in d['hires_paths']] or None
        return cls(images=d['images'], depth=d['depth'], K=d['K'], T_cw=d['T_cw'],
                   session=d['session'], frame_names=[str(x) for x in d['frame_names']],
                   hires_paths=hires, units=str(d['units']), meta=json.loads(str(d['meta'])))


# point-set helpers
@dataclass
class PointSet:
    """Point cloud of one video"""
    xyz: np.ndarray       # (M,3) float32
    rgb: np.ndarray       # (M,3) uint8
    session: int

    def __len__(self):
        return len(self.xyz)


def voxel_downsample(xyz, rgb, voxel):
    """Averages points and colours in each voxel"""
    if len(xyz) == 0:
        return xyz, rgb
    _, inv, counts = np.unique(encode_voxels(xyz, voxel), return_inverse=True, return_counts=True)
    inv = inv.ravel()
    m = len(counts)
    out = np.stack([np.bincount(inv, weights=xyz[:, k], minlength=m) for k in range(3)], 1)
    out /= counts[:, None]
    col = None
    if rgb is not None:
        col = np.stack([np.bincount(inv, weights=rgb[:, k].astype(np.float64), minlength=m)
                        for k in range(3)], 1)
        col = (col / counts[:, None]).astype(np.uint8)
    return out.astype(np.float32), col


def session_points(scene, s, voxel, stride=1):
    """Voxelised point cloud of one video"""
    pts, cols = [], []
    for i in scene.frames_of(s):
        p, c, _ = scene.backproject(i, stride=stride)
        pts.append(p)
        cols.append(c)
    xyz = np.concatenate(pts) if pts else np.zeros((0, 3), np.float32)
    rgb = np.concatenate(cols) if cols else np.zeros((0, 3), np.uint8)
    xyz, rgb = voxel_downsample(xyz, rgb, voxel)
    return PointSet(xyz, rgb, s)


def encode_voxels(xyz, voxel):
    """Voxel indices packed into one int64 key"""
    k = np.floor(xyz / voxel).astype(np.int64) + (1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


_NEIGH = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], np.int64)


def dilate_keys(keys):
    """Adds the 26 neighbours of each voxel key"""
    if len(keys) == 0:
        return keys
    offs = (_NEIGH[:, 0] << 42) + (_NEIGH[:, 1] << 21) + _NEIGH[:, 2]
    return np.unique((keys[:, None] + offs[None, :]).ravel())


def connected_components(xyz, voxel):
    """Labels points by 26-connected voxel components"""
    from scipy import ndimage

    if len(xyz) == 0:
        return np.zeros(0, np.int64)
    idx = np.floor(xyz / voxel).astype(np.int64)
    idx -= idx.min(0)
    shape = idx.max(0) + 1
    if np.prod(shape) > 4e8:   # absurdly large extents: fall back to coarser voxels
        return connected_components(xyz, voxel * 2)
    grid = np.zeros(shape, bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, _ = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    return lab[idx[:, 0], idx[:, 1], idx[:, 2]]


def estimate_normals(P, k=12, return_planarity=False):
    """Surface normals from local PCA"""
    from scipy.spatial import cKDTree

    k = min(k, len(P))
    tree = cKDTree(P)
    normals = np.zeros((len(P), 3), np.float32)
    planarity = np.zeros(len(P), np.float32)
    for a in range(0, len(P), 200_000):           # chunked: real scans have ~1M points
        _, nb = tree.query(P[a:a + 200_000], k=k)
        Q = P[nb] - P[nb].mean(1, keepdims=True)
        w, v = np.linalg.eigh(np.einsum('nki,nkj->nij', Q, Q))
        normals[a:a + 200_000] = v[:, :, 0]
        planarity[a:a + 200_000] = (w[:, 1] - w[:, 0]) / (w[:, 2] + 1e-12)
    return (normals, planarity) if return_planarity else normals
