"""
test_video_plumbing.py - the real video path with VGGT replaced by ground truth

video file -> keyframes -> VGGT sized frames -> scene -> cached detections ->
reasoning -> outputs, all on CPU without downloading any model.
"""
import contextlib
import io

import cv2
import numpy as np

from roomdiff.config import Config
from roomdiff.detect import load_detections, save_detections
from roomdiff.evaluate import evaluate_synthetic
from roomdiff.frames import extract_keyframes
from roomdiff.objects import Detection
from roomdiff.pipeline import analyze
from roomdiff.reconstruct import load_frames, scene_from_predictions
from roomdiff.report import write_outputs
from roomdiff.scene import Scene
from roomdiff.synthetic import make_synthetic


def test_video_to_report(tmp_path):
    scene, dets, gt = make_synthetic(seed=3)
    cfg = Config()
    cfg.frames_per_video, cfg.sample_fps = 16, 1000          # keep every frame of the short clips
    vids = []
    for s in (0, 1):
        v = tmp_path / f's{s}.avi'
        w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*'MJPG'), 4, (640, 480))
        for i in scene.frames_of(s):
            w.write(cv2.cvtColor(cv2.resize(scene.images[i], (640, 480), interpolation=cv2.INTER_NEAREST),
                                 cv2.COLOR_RGB2BGR))
        w.release()
        vids.append(v)
    fb = extract_keyframes(vids[0], tmp_path / 'frames' / 'before', cfg)
    fa = extract_keyframes(vids[1], tmp_path / 'frames' / 'after', cfg)
    assert len(fb) == len(fa) == 16

    paths, session, arr = load_frames(fb, fa)
    S, H, W = arr.shape[:3]
    assert (W, H) == (518, 392) and W % 14 == 0 and H % 14 == 0
    K = scene.K.copy()
    K[:, 0] *= W / 320
    K[:, 1] *= H / 240
    depth = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) for d in scene.depth])
    conf = 5 + np.random.default_rng(0).random(depth.shape)
    rec = scene_from_predictions(paths, session, arr, scene.T_cw[:, :3, :4], K, depth, conf, cfg)
    rec.save(tmp_path / 'scene.npz')
    rec = Scene.load(tmp_path / 'scene.npz')

    dets2 = [Detection(d.frame, d.label, d.score, d.box,
                       cv2.resize(d.mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool),
                       d.embedding) for d in dets]
    save_detections(dets2, tmp_path / 'dets.npz')
    dets3 = load_detections(tmp_path / 'dets.npz')
    assert all((a.mask == b.mask).all() and a.label == b.label for a, b in zip(dets2, dets3))

    with contextlib.redirect_stdout(io.StringIO()):
        res = analyze(rec, dets3, Config())
        ev = evaluate_synthetic(res, gt)
        write_outputs(res, tmp_path / 'out', Config())
    assert ev['passed'] == ev['total'] and not ev['false_positives'], ev
    assert list((tmp_path / 'out' / 'figures').glob('evidence_C*.jpg'))
