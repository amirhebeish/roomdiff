"""
viewer.py - writes the interactive 3D viewer as one self contained html file
"""

import base64
import json
from pathlib import Path

import numpy as np

from ..objects import main_copy, pair_sure
from ..scene import AFTER, BEFORE, voxel_downsample
from .figures import COLORS

HERE = Path(__file__).parent


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _datauri(path, mime='image/jpeg'):
    if path is None or not Path(path).exists():
        return None
    return f'data:{mime};base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def _quant(xyz, bounds):
    """Positions as uint16 inside the scene bounds"""
    lo, hi = np.array(bounds[:3]), np.array(bounds[3:])
    q = np.round((xyz - lo) / np.maximum(hi - lo, 1e-6) * 65535)
    return np.clip(q, 0, 65535).astype('<u2')


def _cloud(xyz, rgb, bounds, verdict=None, max_points=220_000, seed=0):
    """Encoded point cloud for the viewer"""
    if len(xyz) > max_points:
        idx = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
        verdict = verdict[idx] if verdict is not None else None
    d = {'pos': _b64(_quant(xyz, bounds)), 'col': _b64(rgb.astype(np.uint8))}
    if verdict is not None:
        d['verdict'] = _b64(verdict.astype(np.uint8))
    return d


def _box(o):
    b = o.box()
    return {'center': [float(x) for x in b['center']], 'size': [float(x) for x in b['size']], 'yaw': float(b['yaw'])}


def _focus(*objs):
    """Where the camera flies to for a change"""
    pts = np.concatenate([o.points for o in objs if o is not None])
    c = (pts.min(0) + pts.max(0)) / 2
    return {'center': [float(x) for x in c], 'radius': float(np.linalg.norm(pts.max(0) - pts.min(0)) / 2)}


def write_viewer(res, path, images, title, diagnostics, cfg):
    """Writes viewer.html with the data and the bundled viewer inlined"""
    ps, ev = res.ps, res.ev
    allp = np.concatenate([ps[BEFORE].xyz, ps[AFTER].xyz])
    lo = np.percentile(allp, 0.2, 0) - 0.05
    hi = np.percentile(allp, 99.8, 0) + 0.05
    bounds = [float(x) for x in np.concatenate([lo, hi])]

    clouds = {}
    for s, name in ((BEFORE, 'before'), (AFTER, 'after')):
        xyz, rgb, ver = ps[s].xyz, ps[s].rgb, ev[s].verdict
        keep = np.all((xyz >= lo) & (xyz <= hi), 1)
        clouds[name] = _cloud(xyz[keep], rgb[keep], bounds, ver[keep], seed=s)

    def pts(o):
        if o is None:
            return None
        xyz, rgb = voxel_downsample(o.points, o.colors, 0.03)
        return _cloud(xyz, rgb, bounds, max_points=20000)

    changes = []
    for c in res.cs.changes:
        changes.append({
            'id': c.id, 'kind': c.kind, 'label': c.label, 'sentence': c.sentence,
            'confidence': round(float(c.confidence), 2), 'color': COLORS[c.kind],
            'before': _box(c.before) if c.before is not None else None,
            'after': _box(c.after) if c.after is not None else None,
            'pts_before': pts(c.before), 'pts_after': pts(c.after),
            'image': _datauri(images.get(c.id)), 'focus': _focus(c.before, c.after),
            'motion': ({'pivot': [float(x) for x in c.before.center[:2]],
                        'disp': [float(x) for x in c.displacement],
                        'yaw': float(c.yaw_deg or 0.0)}
                       if c.kind in ('moved', 'likely_moved') and c.displacement else None),
        })
    unverified = []
    for k, (o, a) in enumerate(res.cs.unverified):
        unverified.append({'id': f'U{k + 1}', 'label': o.label, 'box': _box(o),
                           'sentence': res.describer.describe_unverified(o, a),
                           'image': _datauri(images.get(f'U{k + 1}')), 'focus': _focus(o)})
    unchanged = [{'label': main_copy(a, b).label, 'box': _box(a or b)} for a, b in res.cs.unchanged
                 if pair_sure(a, b)]

    cams = {}
    for s, name in ((BEFORE, 'before'), (AFTER, 'after')):
        lst = []
        for i in res.scene.frames_of(s):
            R = res.scene.T_cw[i, :3, :3]
            c = res.scene.cam_centers([i])[0]
            lst.append({'c': c.tolist(), 'r': R[0].tolist(), 'd': R[1].tolist(), 'f': R[2].tolist()})
        cams[name] = lst

    data = {
        'title': title, 'headline': res.describer.headline(), 'bounds': bounds,
        'point_size': float(max(cfg.voxel * 1.6, 0.02)),
        'cut_height': float(min(2.0, np.percentile(allp[:, 2], 97) - 0.15)),
        'clouds': clouds, 'changes': changes, 'unverified': unverified, 'unchanged': unchanged,
        'cameras': cams, 'colors': COLORS, 'diagnostics': diagnostics,
        'surprise': [_datauri(p) for p in images.get('surprise', [])],
        'surprise_best': int(images.get('surprise_best', 0)),
        'images': {'overview': _datauri(images.get('overview'), 'image/png'),
                   'occupancy': _datauri(images.get('occupancy'), 'image/png')},
    }
    html = (HERE / 'viewer_template.html').read_text()
    bundle = (HERE / 'viewer_bundle.js').read_text()
    payload = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    html = html.replace('__TITLE__', title).replace('__DATA__', payload)
    bundle = bundle.replace('</script', '<\\/script')
    html = html.replace('<script>__BUNDLE__</script>', '<script>' + bundle + '</script>')
    path.write_text(html)
    return path
