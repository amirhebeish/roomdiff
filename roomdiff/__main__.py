"""
__main__.py - command line interface

Run: python -m roomdiff run --before before.mp4 --after after.mp4 --out runs/my_room
     python -m roomdiff analyze --out runs/my_room   (re-runs the reasoning on the cached GPU results)
     python -m roomdiff demo --out runs/synthetic     (synthetic room with ground truth, CPU only)
"""

import argparse
from pathlib import Path

from .config import Config


def _cfg(args):
    """Config with the command line overrides applied"""
    cfg = Config()
    if getattr(args, 'frames', None):
        cfg.frames_per_video = args.frames
    if getattr(args, 'camera_height', None):
        cfg.camera_height_m = args.camera_height
    if getattr(args, 'classes', None):
        cfg.classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    if getattr(args, 'extra_classes', None):
        cfg.extra_classes += [c.strip() for c in args.extra_classes.split(',') if c.strip()]
    if getattr(args, 'tau', None):
        cfg.tau_abs = args.tau
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog='roomdiff', description='What changed in the room? 3D change detection '
                                 'from two phone videos.')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--out', required=True, help='output folder')
        p.add_argument('--camera-height', type=float,
                       help='phone height above the floor while filming, metres (default 1.4)')
        p.add_argument('--tau', type=float, help='depth agreement tolerance in metres (default 0.06)')

    r = sub.add_parser('run', help='full pipeline on two videos (GPU recommended)')
    r.add_argument('--before', required=True, help='video (or folder of photos) of the room before')
    r.add_argument('--after', required=True, help='video (or folder of photos) of the room after')
    r.add_argument('--frames', type=int, help='keyframes per video (default 24)')
    r.add_argument('--classes', help='comma-separated detector vocabulary (replaces the default list)')
    r.add_argument('--extra-classes', help='comma-separated classes added to the default list')
    r.add_argument('--no-reuse', action='store_true', help='ignore cached reconstruction/detections')
    common(r)

    a = sub.add_parser('analyze', help='re-run the change reasoning on cached reconstruction + detections')
    common(a)

    d = sub.add_parser('demo', help='synthetic benchmark with ground truth (CPU, ~1 min)')
    d.add_argument('--seed', type=int, default=0)
    common(d)

    args = ap.parse_args(argv)
    cfg = _cfg(args)
    if args.cmd == 'run':
        from .pipeline import run_from_videos
        run_from_videos(args.before, args.after, args.out, cfg, reuse=not args.no_reuse)
    elif args.cmd == 'analyze':
        from .detect import load_detections
        from .pipeline import analyze
        from .report import write_outputs
        from .scene import Scene
        out = Path(args.out)
        scene = Scene.load(out / 'cache' / 'scene.npz')
        dets = load_detections(out / 'cache' / 'detections.npz')
        res = analyze(scene, dets, cfg)
        write_outputs(res, out, cfg)
    elif args.cmd == 'demo':
        from .pipeline import run_synthetic
        run_synthetic(args.out, cfg, seed=args.seed)


if __name__ == '__main__':
    main()
