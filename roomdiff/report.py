"""
report.py - writes the outputs: json, markdown report, figures, maps, point clouds, viewer
"""

import json
from pathlib import Path

import numpy as np

from .objects import main_copy, pair_sure
from .occupancy import save_ros_map
from .scene import AFTER, BEFORE
from .viz.figures import (evidence_image, inventory_items, inventory_sheet, occupancy_figure, overview_map,
                          surprise_frames, unverified_image)
from .viz.viewer import write_viewer
from .visibility import DepthCache


def _obj_json(o):
    if o is None:
        return None
    b = o.box()
    return {'id': o.id, 'label': o.label, 'source': o.source,
            'center_m': [round(float(x), 3) for x in o.center],
            'box': {'center': [round(float(x), 3) for x in b['center']],
                    'size': [round(float(x), 3) for x in b['size']], 'yaw_deg': round(float(b['yaw']), 1)},
            'size_text': o.size_text(), 'n_points': int(len(o.points)),
            'seen_in_frames': [int(f) for f in o.frames]}


def _assess_json(a):
    if a is None:
        return None
    return {'status': a.status, 'fraction_seen_through': round(float(a.frac_changed), 3),
            'point_verdicts': a.counts, 'confirming_views': a.confirming_views,
            'views_that_looked': a.looking_views, 'reason': a.reason}


def write_ply(path, xyz, rgb):
    """Coloured binary ply"""
    with open(path, 'wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        arr = np.zeros(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                        ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        arr['x'], arr['y'], arr['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr['r'], arr['g'], arr['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(arr.tobytes())


def diagnostics(res):
    """Short lines on how the run went"""
    info = res.info
    g = info.get('gravity', {})
    out = []
    if g:
        out.append(f"Floor plane {'found' if g.get('floor_found') else 'NOT found (scale is a rough guess)'}; "
                   f"metric scale from an assumed {g.get('camera_height_assumed', 1.4):.2f} m phone height.")
        if 'manhattan_support' in g:
            out.append(f"Up taken from the room itself (walls, floor, table tops: {g['manhattan_support']*100:.0f}% of "
                       f"surfaces agree); the phone's own tilt was {g['up_prior_vs_floor_deg']:.0f}° off.")
    r = info.get('refine')
    if r and 'inliers_5cm_before' in r:
        out.append(f"Before/after registration refinement {'applied' if r['applied'] else 'not needed / rejected'} "
                   f"(surface agreement {r['inliers_5cm_before']*100:.0f}% → {r['inliers_5cm_after']*100:.0f}%).")
    c = info.get('calibration')
    if c:
        out.append(f"Measured depth noise {c['depth_noise_rel']*100:.1f}% of distance (each video against itself), "
                   f"so the depth tolerance is {c['tau_rel']*100:.1f}%.")
    t = info.get('self_test')
    if t:
        out.append(f"Self-test, video 1 against itself where nothing changed: {t['false_alarm_rate']*100:.2f}% of "
                   f"{t['points_tested']} points falsely flagged.")
    if 'overlap' in info:
        out.append(f"{info['overlap']*100:.0f}% of the 'after' surface lies on 'before' surface.")
    for s, name in ((BEFORE, 'before'), (AFTER, 'after')):
        c = res.ev[s].counts()
        tot = max(1, sum(c.values()))
        out.append(f"'{name}' points judged by the other video: " +
                   ', '.join(f'{k} {v/tot*100:.0f}%' for k, v in c.items()))
    ev = info.get('evaluation')
    if ev:
        out.append(f"Synthetic ground truth: {ev['passed']}/{ev['total']} checks passed, "
                   f"{len(ev['false_positives'])} false positives.")
    return out


def write_outputs(res, out, cfg, title=None):
    """Writes every output of a run"""
    out = Path(out)
    fig_dir = out / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / 'maps').mkdir(exist_ok=True)
    (out / 'pointclouds').mkdir(exist_ok=True)
    title = title or ('Synthetic room · what changed?' if res.scene.meta.get('source') == 'synthetic'
                      else 'What changed in the room?')
    res.info.setdefault('gravity', {})['camera_height_assumed'] = cfg.camera_height_m

    images = {}
    overview_map(res, fig_dir / 'overview.png')
    images['overview'] = fig_dir / 'overview.png'
    occupancy_figure(res, fig_dir / 'occupancy.png')
    images['occupancy'] = fig_dir / 'occupancy.png'
    cache = DepthCache(res.scene)
    from .expect import MemoryRenderer
    mr = MemoryRenderer(res.scene, BEFORE)
    for c in res.cs.changes:
        p = evidence_image(res, c, fig_dir / f'evidence_{c.id}.jpg', cache, mr)
        if p:
            images[c.id] = p
    images['surprise'], images['surprise_best'] = surprise_frames(res, mr, fig_dir / 'expected_vs_actual', cfg)
    images['surprise_gif'] = fig_dir / 'expected_vs_actual.gif'
    for k, (o, _) in enumerate(res.cs.unverified):
        p = unverified_image(res, o, fig_dir / f'cant_tell_U{k + 1}.jpg', cache)
        if p:
            images[f'U{k + 1}'] = p

    if inventory_sheet(res, fig_dir / 'inventory.jpg', cache):
        images['inventory'] = fig_dir / 'inventory.jpg'

    save_ros_map(res.occ.before, res.occ, out / 'maps' / 'map_before')
    save_ros_map(res.occ.after, res.occ, out / 'maps' / 'map_after')
    np.save(out / 'maps' / 'change_layer.npy', res.occ.change)

    write_ply(out / 'pointclouds' / 'before.ply', res.ps[BEFORE].xyz, res.ps[BEFORE].rgb)
    write_ply(out / 'pointclouds' / 'after.ply', res.ps[AFTER].xyz, res.ps[AFTER].rgb)

    diag = diagnostics(res)
    # JSON
    js = {
        'headline': res.describer.headline(),
        'units': res.scene.units,
        'frame': "z up, floor at z=0; x points along the first 'before' camera's view, y to its left",
        'changes': [{
            'id': c.id, 'type': c.kind, 'label': c.label, 'description': c.sentence,
            'confidence': round(float(c.confidence), 3),
            'displacement_m': [round(float(x), 3) for x in c.displacement] if c.displacement else None,
            'rotation_deg': None if c.yaw_deg is None else round(float(c.yaw_deg), 1),
            'before': _obj_json(c.before), 'after': _obj_json(c.after),
            'evidence_before_object': _assess_json(c.before_assess),
            'evidence_after_object': _assess_json(c.after_assess),
            'evidence_image': str(images[c.id].relative_to(out)) if c.id in images else None,
            **({'extra': c.details} if c.details else {}),
        } for c in res.cs.changes],
        'cant_tell': [{'label': o.label, 'description': res.describer.describe_unverified(o, a),
                       'object': _obj_json(o), 'evidence': _assess_json(a)} for o, a in res.cs.unverified],
        'unchanged': [_obj_json(a or b) for a, b in res.cs.unchanged],
        'diagnostics': res.info,
    }
    (out / 'changes.json').write_text(json.dumps(js, indent=2, default=_json_default))

    # Markdown
    L = [f'# {title}', '', f'**{res.describer.headline()}**', '',
         '![floor plan](figures/overview.png)', '', '## Changes', '']
    if not res.cs.changes:
        L.append('No meaningful changes detected.')
    for c in res.cs.changes:
        L += [f"### {c.id} · {c.kind.replace('_', ' ')} · {c.label}  (confidence {c.confidence:.0%})", '',
              c.sentence, '']
        if c.id in images:
            L += [f'![{c.id}]({images[c.id].relative_to(out)})', '']
    if res.cs.unverified:
        L += ["## Can't tell", '']
        for o, a in res.cs.unverified:
            L.append(f'- {res.describer.describe_unverified(o, a)}')
        L.append('')
    L += ['## Expected vs actual', '',
          'For every camera of the second video: what the first video predicts it should see, what it saw, '
          'and the surprise (red = gone, green = new, purple = recoloured, hatched = never seen before).', '',
          '![expected vs actual](figures/expected_vs_actual.gif)', '']
    L += ['## Everything it found', '',
          'Every object the run found, numbered as in the photo sheet. A name with "?" rests on one or two '
          'weak detections; the object is still checked for changes.', '',
          '| # | object | status | where | size |', '|---|---|---|---|---|']
    for k, (o, status, _) in enumerate(inventory_items(res)):
        name = o.label + ('' if o.sure else '?')
        L.append(f'| {k + 1} | {name} | {status} | {res.describer.where(o)} | {o.size_text()} |')
    L += ['']
    if 'inventory' in images:
        L += ['![everything found](figures/inventory.jpg)', '']
    unchanged = sorted({main_copy(a, b).label for a, b in res.cs.unchanged if pair_sure(a, b)})
    L += ['## Unchanged', '', ', '.join(unchanged) or 'none', '',
          '## Navigation view', '', '![occupancy](figures/occupancy.png)', '',
          'ROS map_server files: `maps/map_before.yaml`, `maps/map_after.yaml`.', '',
          '## Diagnostics', ''] + [f'- {d}' for d in diag] + ['']
    (out / 'report.md').write_text('\n'.join(L))

    write_viewer(res, out / 'viewer.html', images, title, diag, cfg)
    print(f"[roomdiff] wrote {out/'viewer.html'}, {out/'report.md'}, {out/'changes.json'}")
    return out


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
