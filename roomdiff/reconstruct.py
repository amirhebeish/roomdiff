"""
reconstruct.py - joint 3D reconstruction of both videos with VGGT

The keyframes of both videos go through VGGT in one forward pass, so every
camera comes out in one shared frame and no separate registration is needed.
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .scene import AFTER, BEFORE, Scene


def _import_vggt():
    """Finds the VGGT clone"""
    try:
        import vggt  # noqa: F401
        return
    except ImportError:
        pass
    here = Path(__file__).resolve().parent.parent
    for cand in [os.environ.get('VGGT_PATH'), here / 'third_party' / 'vggt', Path.cwd() / 'vggt',
                 Path('/content/vggt')]:
        if cand and (Path(cand) / 'vggt' / 'models').exists():
            sys.path.insert(0, str(cand))
            return
    raise ImportError('VGGT not found. Clone it:  git clone https://github.com/facebookresearch/vggt '
                      'third_party/vggt  (or set VGGT_PATH)')


def vggt_size(w, h):
    """Longest side 518 px, both sides multiples of 14, no cropping"""
    if w >= h:
        W = 518
        H = max(14, int(round(h * 518 / w / 14)) * 14)
    else:
        H = 518
        W = max(14, int(round(w * 518 / h / 14)) * 14)
    return W, H


def clean_depth(depth, conf, drop_pct, edge_rel):
    """Removes low confidence pixels and flying pixels at depth edges"""
    d = depth.astype(np.float32).copy()
    thr = np.percentile(conf, drop_pct)
    d[conf < thr] = np.nan
    gy, gx = np.gradient(depth, axis=(1, 2))
    rel = np.sqrt(gx ** 2 + gy ** 2) / np.maximum(depth, 1e-6)
    d[rel > edge_rel] = np.nan
    d[~np.isfinite(d) | (d <= 0)] = np.nan
    return d


def load_frames(frames_before, frames_after):
    """Loads the keyframes of both videos at VGGT size"""
    paths = [Path(p) for p in list(frames_before) + list(frames_after)]
    session = np.array([BEFORE] * len(frames_before) + [AFTER] * len(frames_after))
    ims = [Image.open(p).convert('RGB') for p in paths]
    if len({(im.size[0] >= im.size[1]) for im in ims}) > 1:
        raise ValueError('Both videos must be filmed in the same orientation (both landscape or both portrait).')
    W, H = vggt_size(*ims[0].size)
    arr = np.stack([np.asarray(im.resize((W, H), Image.BICUBIC)) for im in ims])
    return paths, session, arr


def scene_from_predictions(paths, session, arr, extri, intri, depth, conf, cfg, model_name=''):
    """Scene from the VGGT outputs (numpy only, so it can be tested on CPU)"""
    T = np.tile(np.eye(4), (len(paths), 1, 1))
    T[:, :3, :4] = extri
    d = clean_depth(depth, conf, cfg.conf_drop_percentile, cfg.edge_rel_thresh)
    names = [f"{'before' if s == BEFORE else 'after'}_{Path(p).stem}" for s, p in zip(session, paths)]
    return Scene(images=arr, depth=d, K=np.asarray(intri, np.float64), T_cw=T.astype(np.float64),
                 session=np.asarray(session), frame_names=names, hires_paths=[str(p) for p in paths],
                 meta={'source': 'vggt', 'model': model_name, 'input_size': [arr.shape[2], arr.shape[1]]})


def run_vggt(arr, model_name):
    """VGGT forward pass as in the official README"""
    import torch
    _import_vggt()
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
        print('[roomdiff] WARNING: no GPU found, VGGT on CPU will be very slow')
    try:
        model = VGGT.from_pretrained(model_name)
    except Exception:  # fall back to the plain state dict the VGGT demos use
        model = VGGT()
        url = f'https://huggingface.co/{model_name}/resolve/main/model.pt'
        model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location='cpu'))
    model = model.to(device).eval()

    S, H, W = arr.shape[:3]
    left = 0
    if H > W:
        # portrait frames: VGGT is trained on landscape and square images and predicts the same
        # field of view on both axes for a tall image (fx != fy), so pad to a square like its
        # own pad mode (white) and crop the outputs back afterwards
        left = (H - W) // 2
        sq = np.full((S, H, H, 3), 255, np.uint8)
        sq[:, :, left:left + W] = arr
        arr = sq
    images = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255.0).to(device)[None]   # (1,S,3,H,W) in [0,1]
    try:
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=dtype, enabled=device == 'cuda'):
                tokens, ps_idx = model.aggregator(images)
            pose_enc = model.camera_head(tokens)[-1]
            extri, intri = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth, conf = model.depth_head(tokens, images, ps_idx)
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(f'GPU out of memory with {arr.shape[0]} frames: re-run with fewer keyframes, '
                           'e.g. --frames 16') from e
    extri = extri[0].float().cpu().numpy()
    intri = intri[0].float().cpu().numpy()
    depth = depth[0, ..., 0].float().cpu().numpy()
    conf = conf[0].float().cpu().numpy()
    if left:
        depth = depth[:, :, left:left + W]
        conf = conf[:, :, left:left + W]
        intri[:, 0, 2] -= left
    # square pixels: fx and fy should agree, anything else means distorted geometry
    ratio = float(np.median(intri[:, 0, 0] / intri[:, 1, 1]))
    print(f'[roomdiff]   focal fx/fy = {ratio:.2f}' + ('' if abs(ratio - 1) < 0.1 else '  WARNING: expected 1.0'))
    out = (extri, intri, depth, conf)
    del model, tokens
    if device == 'cuda':
        torch.cuda.empty_cache()
    return out


def reconstruct_vggt(frames_before, frames_after, cfg):
    """Keyframes of both videos -> shared scene"""
    paths, session, arr = load_frames(frames_before, frames_after)
    extri, intri, depth, conf = run_vggt(arr, cfg.vggt_model)
    return scene_from_predictions(paths, session, arr, extri, intri, depth, conf, cfg, cfg.vggt_model)
