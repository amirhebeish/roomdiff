"""
pipeline.py - runs the full pipeline: videos -> shared 3D scene -> changes -> outputs
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .align import gravity_align, overlap_report, refine_after_onto_before
from .appearance import find_appearance_changes
from .changes import Change, ChangeSet, detect_changes
from .config import Config
from .describe import Describer, describe_all
from .objects import fuse_objects, label_from_detections
from .occupancy import Occupancy, build_occupancy
from .scene import AFTER, BEFORE, Scene, session_points
from .visibility import DepthCache, evaluate


@dataclass
class Result:
    """Everything one run produced"""
    scene: Scene
    ps: dict                  # session -> PointSet
    ev: dict                  # session -> Evidence (that session's points judged by the other's cameras)
    cs: ChangeSet
    describer: Describer
    occ: Occupancy
    info: dict = field(default_factory=dict)


def log(msg):
    print(f"[roomdiff {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def analyze(scene, detections, cfg, region_namer=None):
    """Alignment, calibration, objects, visibility test and change reasoning"""
    info = {}
    log('aligning to gravity / floor, estimating metric scale')
    info['gravity'] = gravity_align(scene, cfg.camera_height_m)
    if not info['gravity']['floor_found']:
        log('  ! no floor plane found: scale is a rough guess')
    if cfg.refine_sessions:
        info['refine'] = refine_after_onto_before(scene)
        r = info['refine']
        log(f"  before/after refinement: applied={r['applied']}, surface agreement within 5 cm "
            f"{r.get('inliers_5cm_before', 0)*100:.0f}% -> {r.get('inliers_5cm_after', 0)*100:.0f}%")
    info['overlap'] = overlap_report(scene)
    log(f"  {info['overlap']*100:.0f}% of the 'after' surface coincides with 'before' surface")
    if info['overlap'] < 0.3:
        log('  note: little shared surface (very different viewpoints, or registration trouble); '
            "regions seen by only one recording will be reported as 'can't tell'")

    ps = {s: session_points(scene, s, cfg.voxel) for s in (BEFORE, AFTER)}
    log(f'  point clouds: before {len(ps[BEFORE])}, after {len(ps[AFTER])} points')
    if cfg.auto_calibrate:
        from .calibrate import calibrate, self_test
        info['calibration'] = calibrate(scene, ps, cfg)
        c = info['calibration']
        log(f"  measured depth noise {c['depth_noise_rel']*100:.1f}% -> tolerance {c['tau_rel']*100:.1f}% of distance")
        info['self_test'] = self_test(scene, cfg)
        if info['self_test']:
            rate = info['self_test']['false_alarm_rate'] * 100
            log(f'  self-test (video 1 against itself): {rate:.2f}% of points falsely flagged')

    log('fusing 2D detections into 3D objects')
    objs = {BEFORE: fuse_objects(scene, detections, BEFORE, cfg, 'A'),
            AFTER: fuse_objects(scene, detections, AFTER, cfg, 'B')}
    log(f'  objects: before {len(objs[BEFORE])}, after {len(objs[AFTER])}')

    log('visibility reasoning (free-space evidence across recordings)')
    cache = DepthCache(scene)
    ev = {BEFORE: evaluate(ps[BEFORE].xyz, scene, scene.frames_of(AFTER), cfg, cache),
          AFTER: evaluate(ps[AFTER].xyz, scene, scene.frames_of(BEFORE), cfg, cache)}
    for s, name in ((BEFORE, 'before'), (AFTER, 'after')):
        c = ev[s].counts()
        tot = max(1, sum(c.values()))
        log('  ' + name + ': ' + ', '.join(f'{k} {v/tot*100:.0f}%' for k, v in c.items()))

    log('object-level change reasoning')
    cs = detect_changes(scene, objs[BEFORE], objs[AFTER], ps[BEFORE], ps[AFTER], ev[BEFORE], ev[AFTER], cfg, detections)
    if cfg.appearance:
        add_appearance_changes(cs, ps, ev, cfg, info)
    # unlabelled regions: vote with the 2D detections that cover them in their own video
    cache = DepthCache(scene)
    for c in cs.changes:
        if c.label != 'unidentified object':
            continue
        for o in (c.before, c.after):
            if o is not None and o.source == 'region':
                lab = label_from_detections(scene, o, detections, cache)
                if lab:
                    c.label = lab
                    break
    if region_namer is not None:
        for c in cs.changes:
            if c.label != 'unidentified object' and c.kind != 'appearance':
                continue
            o = c.after if c.kind == 'added' else c.before
            if c.kind == 'appearance':
                o = c.after if c.details.get('direction') == 'added' else c.before
            if o is not None and o.source in ('region', 'appearance'):
                guess = region_namer(scene, o)
                if guess:
                    c.details['appearance_guess'] = guess
    describer = describe_all(cs)
    for c in cs.changes:
        g = c.details.get('appearance_guess')
        if g:
            c.sentence += f' (It looks like it could be {g}.)'
    occ = build_occupancy(ps, ev, cfg)
    log(describer.headline())
    for c in cs.changes:
        log(f'  [{c.id}] {c.kind:<12} {c.sentence}')
    return Result(scene, ps, ev, cs, describer, occ, info)


def run_from_videos(before_video, after_video, out_dir, cfg=None,
                    reuse=True):
    """Full pipeline on two videos (needs a GPU)"""
    from .frames import extract_keyframes
    cfg = cfg or Config()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene_path, det_path = out / 'cache' / 'scene.npz', out / 'cache' / 'detections.npz'
    (out / 'cache').mkdir(exist_ok=True)

    if reuse and scene_path.exists():
        log(f'reusing {scene_path}')
        scene = Scene.load(scene_path)
    else:
        from .reconstruct import reconstruct_vggt
        log('extracting keyframes')
        fb = extract_keyframes(before_video, out / 'frames' / 'before', cfg)
        fa = extract_keyframes(after_video, out / 'frames' / 'after', cfg)
        log(f'  {len(fb)} + {len(fa)} keyframes')
        log('joint 3D reconstruction of both recordings with VGGT')
        scene = reconstruct_vggt(fb, fa, cfg)
        scene.save(scene_path)

    if reuse and det_path.exists():
        log(f'reusing {det_path}')
        from .detect import load_detections
        dets = load_detections(det_path)
    else:
        from .detect import detect_scene, save_detections
        log('open-vocabulary detection + segmentation (Grounding DINO + SAM + CLIP)')
        dets = detect_scene(scene, cfg)
        save_detections(dets, det_path)
        log(f'  {len(dets)} detections')

    namer = None
    try:
        from .detect import ClipRegionNamer
        namer = ClipRegionNamer(cfg)
    except Exception as e:            # CLIP is optional for naming unidentified regions
        log(f'  (region naming disabled: {e})')
    res = analyze(scene, dets, cfg, region_namer=namer)
    from .report import write_outputs
    write_outputs(res, out, cfg)
    return res


def run_synthetic(out_dir, cfg=None, seed=0):
    """Full pipeline on the synthetic room, scored against ground truth"""
    from .report import write_outputs
    from .synthetic import make_synthetic
    cfg = cfg or Config()
    log('rendering synthetic before/after recordings')
    scene, dets, gt = make_synthetic(seed=seed)
    res = analyze(scene, dets, cfg)
    from .evaluate import evaluate_synthetic
    res.info['evaluation'] = evaluate_synthetic(res, gt)
    write_outputs(res, Path(out_dir), cfg)
    return res


def add_appearance_changes(cs, ps, ev, cfg, info):
    """Adds the colour-only changes to the change set"""
    changed = [o for c in cs.changes for o in (c.before, c.after) if o is not None]
    found, stats = find_appearance_changes(ps, ev, changed, cfg)
    info['appearance'] = stats
    for a, b, det in found:
        conf = float(np.clip(det['colour_difference'] / 45, 0.3, 0.85))
        cs.changes.append(Change('', 'appearance', 'flat item', conf, a, b, details=det))
    order = {'moved': 0, 'likely_moved': 1, 'added': 2, 'removed': 3, 'appearance': 4}
    cs.changes.sort(key=lambda c: (order[c.kind], -c.confidence))
    for k, c in enumerate(cs.changes):
        c.id = f'C{k + 1}'
