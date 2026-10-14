"""
frames.py - picks sharp, evenly spread keyframes from a video
"""

from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _sharpness(bgr):
    """Variance of the Laplacian, higher is sharper"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (320, int(320 * g.shape[0] / g.shape[1])))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _resize_max(bgr, max_side):
    """Resize so the longest side is at most max_side"""
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return bgr
    return cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def extract_keyframes(src, out_dir, cfg):
    """Sharpest frame in each of n equal time bins, written as jpg"""
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_out = cfg.frames_per_video

    if src.is_dir():   # a folder of photos works too
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
        if not files:
            raise FileNotFoundError(f'no images in {src}')
        pick = np.linspace(0, len(files) - 1, min(n_out, len(files))).round().astype(int)
        out = []
        for k, i in enumerate(pick):
            im = _resize_max(cv2.imread(str(files[i])), cfg.hires_max_side)
            p = out_dir / f'frame_{k:03d}.jpg'
            cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out.append(p)
        return out

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f'cannot open video {src}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / cfg.sample_fps)))
    cands = []                                   # (frame index, sharpness)
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, fr = cap.retrieve()
            if ok:
                cands.append((idx, _sharpness(fr)))
        idx += 1
    cap.release()
    if not cands:
        raise IOError(f'no frames decoded from {src}')

    bins = np.array_split(np.arange(len(cands)), min(n_out, len(cands)))
    chosen = sorted(cands[b[np.argmax([cands[j][1] for j in b])]][0] for b in bins if len(b))

    out, want = [], set(chosen)
    cap = cv2.VideoCapture(str(src))
    idx = 0
    while want:
        ok = cap.grab()
        if not ok:
            break
        if idx in want:
            ok, fr = cap.retrieve()
            if ok:
                p = out_dir / f'frame_{len(out):03d}.jpg'
                cv2.imwrite(str(p), _resize_max(fr, cfg.hires_max_side), [cv2.IMWRITE_JPEG_QUALITY, 95])
                out.append(p)
            want.discard(idx)
        idx += 1
    cap.release()
    return out
