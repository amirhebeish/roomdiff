"""
evaluate.py - scores a synthetic run against its ground truth
"""

import numpy as np

from .align import umeyama
from .scene import BEFORE


def _room_to_scene(res, gt):
    """Maps ground truth room coordinates into the final scene frame"""
    T = gt['T_cw_room']
    idx = res.scene.frames_of(BEFORE)
    C_room = -np.einsum('nji,nj->ni', T[idx, :3, :3], T[idx, :3, 3])
    C_scene = res.scene.cam_centers(idx)
    c, R, t = umeyama(C_room, C_scene)
    return lambda xy, z=0.3: c * R @ np.array([xy[0], xy[1], z]) + t


def evaluate_synthetic(res, gt, tol=0.45):
    """Prints and returns the scorecard"""
    f = _room_to_scene(res, gt)
    truth = gt['truth']
    cs = res.cs
    used = set()
    rows = []

    def near(o, xy):
        return o is not None and np.linalg.norm(o.center[:2] - f(xy)[:2]) < tol

    def find(kind, label, xy_before=None, xy_after=None):
        for c in cs.changes:
            if c.id in used or c.kind != kind or c.label != label:
                continue
            if xy_before is not None and not near(c.before, xy_before):
                continue
            if xy_after is not None and not near(c.after, xy_after):
                continue
            used.add(c.id)
            return c
        return None

    for label, b, a, yaw in truth['moved']:
        c = find('moved', label, b, a)
        note = ''
        if c is not None:
            true_d = f(a)[:2] - f(b)[:2]
            err = float(np.linalg.norm(np.array(c.displacement[:2]) - true_d))
            note = f'displacement error {err*100:.0f} cm of {np.linalg.norm(true_d):.2f} m'
            if c.yaw_deg is not None:
                note += f', rotation {abs(c.yaw_deg):.0f} deg (true {yaw:.0f})'
        rows.append(('moved', label, c is not None, c.id if c else '-', note))
    for label, xy in truth['removed']:
        c = find('removed', label, xy_before=xy)
        rows.append(('removed', label, c is not None, c.id if c else '-', ''))
    for label, xy in truth['added']:
        c = find('added', label, xy_after=xy)
        rows.append(('added', label, c is not None, c.id if c else '-', ''))
    for label, xy in truth['unverified']:
        hit = [(o, a) for o, a in cs.unverified if o.label == label and near(o, xy)]
        rows.append(("can't tell", label, bool(hit), hit[0][0].id if hit else '-',
                     hit[0][1].reason if hit else ''))
    for label, xy, direction in truth.get('appearance', []):
        hit = None
        for c in cs.changes:
            if c.id not in used and c.kind == 'appearance' and near(c.before, xy):
                hit = c
                used.add(c.id)
                break
        note = f"direction: {hit.details.get('direction')} (true {direction})" if hit else ''
        rows.append(('looks diff.', label, hit is not None, hit.id if hit else '-', note))
    unchanged_labels = {(a or b).label for a, b in cs.unchanged}
    cant_tell_labels = {o.label for o, _ in cs.unverified}
    changed_labels = {c.label for c in cs.changes}
    for label in truth['unchanged']:
        # pass = not reported as a change; "can't tell" is an honest answer when one
        # recording barely saw the object
        ok = label not in changed_labels and (label in unchanged_labels or label in cant_tell_labels)
        note = '' if label in unchanged_labels else ("reported as can't tell" if ok else '')
        rows.append(('unchanged', label, ok, '-', note))
    false_pos = [c for c in cs.changes if c.id not in used]
    passed = sum(r[2] for r in rows)
    out = {'rows': [dict(zip(('expected', 'label', 'ok', 'id', 'note'), r)) for r in rows],
           'passed': int(passed), 'total': len(rows),
           'false_positives': [f'{c.id} {c.kind} {c.label}' for c in false_pos]}
    print('\nSynthetic benchmark')
    print(f"{'expected':<12}{'label':<22}{'result':<8}{'id':<6}note")
    for r in rows:
        print(f"{r[0]:<12}{r[1]:<22}{'PASS' if r[2] else 'FAIL':<8}{r[3]:<6}{r[4]}")
    print(f"{passed}/{len(rows)} passed, {len(false_pos)} false positive(s) {out['false_positives']}\n")
    return out
