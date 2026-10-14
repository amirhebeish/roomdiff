"""
occupancy.py - before/after occupancy maps for navigation, exported for ROS map_server
"""

from dataclasses import dataclass

import numpy as np

from .scene import AFTER, BEFORE
from .visibility import CHANGED, KEPT

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
# change layer codes
C_UNKNOWN, C_FREE, C_STATIC, C_NEW_OBSTACLE, C_CLEARED, C_ONE_SIDED = 0, 1, 2, 3, 4, 5


@dataclass
class Occupancy:
    """Before, after and change grids"""
    origin: tuple      # world (x, y) of cell [0, 0] corner
    res: float
    before: np.ndarray  # (ny, nx) UNKNOWN / FREE / OCCUPIED
    after: np.ndarray
    change: np.ndarray  # change layer codes

    def extent(self):
        ny, nx = self.before.shape
        x0, y0 = self.origin
        return [x0, x0 + nx * self.res, y0, y0 + ny * self.res]


def _raster(xyz, origin, res, shape, min_count=1):
    """Cells hit by at least min_count points"""
    ij = np.floor((xyz[:, :2] - origin) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
    g = np.zeros(shape, np.int32)
    np.add.at(g, (ij[ok, 1], ij[ok, 0]), 1)
    return g >= min_count


def build_occupancy(ps, ev, cfg):
    """Occupancy at robot height, coloured by the visibility verdicts"""
    res = cfg.grid_res
    allp = np.concatenate([ps[BEFORE].xyz, ps[AFTER].xyz])
    lo = np.percentile(allp[:, :2], 0.5, 0) - 0.2
    hi = np.percentile(allp[:, :2], 99.5, 0) + 0.2
    shape = (int(np.ceil((hi[1] - lo[1]) / res)), int(np.ceil((hi[0] - lo[0]) / res)))
    z0, z1 = cfg.robot_band
    maps = {}
    for s in (BEFORE, AFTER):
        P = ps[s].xyz
        band = (P[:, 2] > z0) & (P[:, 2] < z1)
        floor = P[:, 2] <= z0
        occ = _raster(P[band], lo, res, shape, min_count=2)
        obs = _raster(P[floor], lo, res, shape) | occ
        m = np.full(shape, UNKNOWN, np.uint8)
        m[obs] = FREE
        m[occ] = OCCUPIED
        maps[s] = m
    ch = np.full(shape, C_UNKNOWN, np.uint8)
    both = (maps[BEFORE] != UNKNOWN) & (maps[AFTER] != UNKNOWN)
    ch[both] = C_FREE
    ch[(maps[BEFORE] != UNKNOWN) ^ (maps[AFTER] != UNKNOWN)] = C_ONE_SIDED
    static = _raster(ps[BEFORE].xyz[(ev[BEFORE].verdict == KEPT) & (ps[BEFORE].xyz[:, 2] > z0)
                                    & (ps[BEFORE].xyz[:, 2] < z1)], lo, res, shape)
    ch[static | ((maps[BEFORE] == OCCUPIED) & (maps[AFTER] == OCCUPIED))] = C_STATIC
    for s, code in ((BEFORE, C_CLEARED), (AFTER, C_NEW_OBSTACLE)):
        P = ps[s].xyz
        sel = (ev[s].verdict == CHANGED) & (P[:, 2] > z0) & (P[:, 2] < z1)
        ch[_raster(P[sel], lo, res, shape, min_count=2)] = code
    return Occupancy(tuple(lo), res, maps[BEFORE], maps[AFTER], ch)


def save_ros_map(grid, occ, path_stem):
    """Writes a map_server pgm + yaml pair"""
    img = np.full(grid.shape, 205, np.uint8)       # unknown
    img[grid == FREE] = 254
    img[grid == OCCUPIED] = 0
    img = img[::-1]                                # PGM rows go top-down, map y goes up
    pgm = path_stem.with_suffix('.pgm')
    with open(pgm, 'wb') as f:
        f.write(f'P5\n{img.shape[1]} {img.shape[0]}\n255\n'.encode())
        f.write(img.tobytes())
    yaml = path_stem.with_suffix('.yaml')
    yaml.write_text(
        f'image: {pgm.name}\nresolution: {occ.res}\norigin: [{occ.origin[0]:.3f}, {occ.origin[1]:.3f}, 0.0]\n'
        'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
