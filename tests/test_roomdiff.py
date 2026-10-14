"""
test_roomdiff.py - geometry checks and the synthetic benchmark (CPU, about 3 min)
"""
import contextlib
import io

import numpy as np
import pytest

from roomdiff.config import Config
from roomdiff.evaluate import evaluate_synthetic
from roomdiff.pipeline import analyze
from roomdiff.scene import Scene
from roomdiff.synthetic import make_synthetic, random_rotation
from roomdiff.visibility import FREE, OCC, OUT, SAME, ray_check


def _wall_scene(dist=3.0):
    """One camera looking down +z at a flat wall `dist` metres away"""
    H, W = 60, 80
    K = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])
    return Scene(images=np.zeros((1, H, W, 3), np.uint8), depth=np.full((1, H, W), dist, np.float32),
                 K=K[None], T_cw=np.eye(4)[None], session=np.array([0]))


def test_ray_check_codes():
    s = _wall_scene(3.0)
    pts = np.array([[0, 0, 3.0],     # on the wall          -> SAME
                    [0, 0, 1.5],     # in front of the wall -> the camera saw through it: FREE
                    [0, 0, 4.5],     # behind the wall      -> hidden: OCC
                    [0, 0, -1.0]])   # behind the camera    -> OUT
    codes = ray_check(pts, s, [0])[0]
    assert list(codes) == [SAME, FREE, OCC, OUT]


def test_similarity_transform_moves_backprojected_points():
    s = _wall_scene(2.0)
    p0, _, _ = s.backproject(0, stride=7)
    rng = np.random.default_rng(1)
    R, t, c = random_rotation(rng), rng.normal(size=3), 0.7
    s.apply_similarity(c, R, t)
    p1, _, _ = s.backproject(0, stride=7)
    assert np.allclose(p1, c * p0 @ R.T + t, atol=1e-4)


def _run(**kw):
    scene, dets, gt = make_synthetic(**kw)
    with contextlib.redirect_stdout(io.StringIO()):
        res = analyze(scene, dets, Config())
        ev = evaluate_synthetic(res, gt)
    return res, ev


@pytest.mark.parametrize('seed', [0, 2])
def test_synthetic_benchmark(seed):
    res, ev = _run(seed=seed)
    failed = [r for r in ev['rows'] if not r['ok']]
    assert not failed, failed
    assert not ev['false_positives']
    moved = [c for c in res.cs.changes if c.kind == 'moved'][0]
    assert 1.4 < np.linalg.norm(moved.displacement[:2]) < 2.1     # true: 1.8 m


def test_no_change_means_no_changes():
    res, ev = _run(seed=4, with_changes=False)
    assert res.cs.changes == []


def test_outputs(tmp_path):
    from roomdiff.pipeline import run_synthetic
    with contextlib.redirect_stdout(io.StringIO()):
        run_synthetic(str(tmp_path))
    for f in ['viewer.html', 'report.md', 'changes.json', 'figures/overview.png',
              'figures/expected_vs_actual.gif', 'figures/expected_vs_actual/surprise_00.jpg',
              'maps/map_after.pgm', 'maps/map_after.yaml', 'pointclouds/before.ply']:
        assert (tmp_path / f).exists(), f
    html = (tmp_path / 'viewer.html').read_text()
    assert 'window.ROOMDIFF' in html and '__BUNDLE__' not in html


def test_memory_render_matches_reality_where_nothing_changed():
    """Rendering video 1 from a video 2 camera must reproduce that camera's depth on unchanged surfaces"""
    from roomdiff.expect import MemoryRenderer
    from roomdiff.scene import AFTER, BEFORE
    scene, _, _ = make_synthetic(seed=1, with_changes=False, noise=False)
    mr = MemoryRenderer(scene, BEFORE)
    agree = []
    for j in scene.frames_of(AFTER):
        _, depth, _ = mr.render(j)
        both = np.isfinite(depth) & np.isfinite(scene.depth[j])
        if both.mean() < 0.2:
            continue
        rel = np.abs(depth[both] - scene.depth[j][both]) / scene.depth[j][both]
        agree.append((rel < 0.03).mean())
    # most remembered pixels match reality; the rest are opposite faces of furniture
    assert len(agree) >= 8 and np.median(agree) > 0.6, agree


def test_up_from_room_geometry():
    # a tilted box room: floor, ceiling and four walls, rotated 15 degrees off the prior
    from roomdiff.align import _rotvec, manhattan_up
    rng = np.random.default_rng(0)
    faces = []
    for axis in range(3):
        for side in (0.0, 3.0):
            p = rng.uniform(0, 3, (3000, 3))
            p[:, axis] = side + rng.normal(0, 0.003, 3000)
            faces.append(p)
    P = np.concatenate(faces)
    R = _rotvec(np.deg2rad(15) * np.array([0.6, 0.8, 0.0]))
    true_up = R @ np.array([0, 0, 1.0])
    up, support = manhattan_up(P @ R.T, prior=np.array([0, 0, 1.0]))
    assert np.degrees(np.arccos(np.clip(up @ true_up, -1, 1))) < 1.0
    assert support > 0.8


def test_name_that_fits_the_size_wins():
    from roomdiff.objects import plausible_label
    rng = np.random.default_rng(0)
    long_thing = rng.uniform([0, 0, 0], [1.6, 0.9, 0.7], (2000, 3))
    assert plausible_label({'chair': 3.4, 'sofa': 3.1}, long_thing) == 'sofa'
    small_thing = rng.uniform([0, 0, 0], [0.5, 0.5, 0.9], (2000, 3))
    assert plausible_label({'chair': 3.4, 'sofa': 3.1}, small_thing) == 'chair'
