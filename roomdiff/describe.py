"""
describe.py - plain English descriptions built from the measured geometry

Every sentence is filled in from measured quantities (distance, direction,
supporting surface, nearest unchanged landmark, number of confirming views) so
the text cannot say anything the geometry did not measure. Directions are as
seen by someone standing where the first video started and looking at the
object, so they stay right for things behind the camera too.
"""

import numpy as np

from .config import LANDMARK_CLASSES
from .objects import main_copy

# things other things are put on
SURFACES = {'table', 'desk', 'sofa', 'bed', 'cabinet', 'shelf', 'chair', 'stool', 'box'}
# big furniture that names the side of something it faces
FACING = {'sofa', 'bed', 'desk', 'TV', 'cabinet', 'shelf', 'bookshelf', 'wardrobe', 'fridge',
          'washing machine', 'oven', 'window', 'door'}


def _article(label):
    return 'an' if label[:1] in 'aeiou' else 'a'


def view_axes(p):
    """Forward and left unit vectors, looking from where the first video started towards p"""
    p = np.asarray(p, float)[:2]
    n = np.linalg.norm(p)
    fwd = p / n if n > 0.3 else np.array([1.0, 0.0])
    return fwd, np.array([-fwd[1], fwd[0]])


def in_view(v, p):
    """A floor vector as (further away, to the left), as seen looking at p from the start"""
    fwd, left = view_axes(p)
    v = np.asarray(v, float)[:2]
    return float(v @ fwd), float(v @ left)


def direction_phrase(dx, dy, min_part=0.12):
    """Direction in words, e.g. 1.2 m to the left and 0.4 m further away"""
    parts = []
    if abs(dy) >= min_part:
        parts.append(f"{abs(dy):.1f} m to the {'left' if dy > 0 else 'right'}")
    if abs(dx) >= min_part:
        parts.append(f"{abs(dx):.1f} m {'further away' if dx > 0 else 'closer'}")
    return ' and '.join(parts) if parts else 'a short distance'


def _box_xy_distance(p, o):
    """Distance from a point to the footprint of an object"""
    lo, hi = np.percentile(o.points[:, :2], 3, 0), np.percentile(o.points[:, :2], 97, 0)
    q = np.clip(p[:2], lo, hi)
    return float(np.linalg.norm(p[:2] - q)), lo, hi


class Describer:
    """Builds the sentences, using unchanged objects as landmarks"""
    def __init__(self, cs):
        self.cs = cs
        lm = []
        for a, b in cs.unchanged:
            # the more complete of the two views of it
            o = main_copy(a, b)
            if o.source == 'detected' and o.sure:
                lm.append(o)
        # prefer furniture-sized landmarks
        self.landmarks = sorted(lm, key=lambda o: (o.label not in LANDMARK_CLASSES, -len(o.points)))

    # spatial words
    def support(self, o):
        """What the object stands on"""
        bottom = o.bottom
        if bottom < 0.15:
            return 'on the floor'
        c = o.center
        best = None
        # unchanged landmarks first, then furniture of the same recording (a bag
        # can sit on a sofa that itself moved)
        same = [x for x in self.cs.objects.get(o.session, [])
                if x is not o and x.source == 'detected' and x.sure and x.label in SURFACES]
        for lm in self.landmarks + sorted(same, key=lambda x: -len(x.points)):
            if lm is o or lm.label not in SURFACES:
                continue
            d, lo, hi = _box_xy_distance(c, lm)
            if d < 0.08 and min(abs(lm.top - bottom), abs(lm.surface - bottom)) < 0.12:
                best = lm
                break
        if best is not None:
            return f'on the {best.label}'
        # its lower part may simply not have been seen (hidden behind something),
        # so don't claim it floats above the floor
        return ''

    def near(self, o, exclude=(), sides=False):
        """Position relative to the nearest landmark"""
        c = o.center
        cands = []
        for lm in self.landmarks:
            if lm in exclude or lm.label == o.label and np.linalg.norm(lm.center[:2] - c[:2]) < 0.3:
                continue
            d, lo, hi = _box_xy_distance(c, lm)
            if d < 0.08 and min(abs(lm.top - o.bottom), abs(lm.surface - o.bottom)) < 0.12:
                continue            # that's the support surface, already mentioned
            cands.append((d, lm))
        if not cands:
            return f'at ({c[0]:.1f}, {c[1]:.1f}) m in the room frame'
        d, lm = min(cands, key=lambda t: t[0])
        name = ('the other ' if lm.label == o.label else 'the ') + lm.label
        v = c[:2] - lm.center[:2]
        f, s = in_view(v, lm.center)
        if d < 0.6:
            if not sides:
                return f'next to {name}'
            # a side is easiest to picture by the furniture it faces
            facing = self.facing(lm, v, exclude=(o.label,))
            if facing is not None:
                return f'at the {facing.label} side of {name}'
            if abs(s) >= abs(f):
                side = 'left' if s > 0 else 'right'
            else:
                side = 'far' if f > 0 else 'near'
            return f'at the {side} side of {name}'
        if abs(s) >= abs(f):
            rel = 'left of' if s > 0 else 'right of'
        else:
            rel = 'beyond' if f > 0 else 'in front of'
        return f'{d:.1f} m {rel} {name}'

    def facing(self, lm, v, exclude=(), max_dist=3.0):
        """The nearest big unchanged furniture lying in direction v from lm, if one clearly does"""
        u = v / (np.linalg.norm(v) + 1e-9)
        best = None
        for x in self.landmarks:
            if x is lm or x.label == lm.label or x.label in exclude or x.label not in FACING:
                continue
            w = x.center[:2] - lm.center[:2]
            dist = np.linalg.norm(w)
            # within 35 degrees of that side, and not further than a room away
            if dist < max_dist and w @ u > np.cos(np.radians(35)) * dist:
                if best is None or dist < best[0]:
                    best = (dist, x)
        return None if best is None else best[1]

    def where(self, o, exclude=(), sides=False):
        """Support and position together"""
        s = self.support(o)
        if s.startswith('on the ') and s != 'on the floor' and not sides:
            return s
        n = self.near(o, exclude, sides)
        if s.startswith('on the ') and s != 'on the floor':
            return f'{s}, {n}'
        return f'{s}, {n}' if s else n

    def from_to(self, a, b):
        """Old and new place, with more detail when the short form says the same twice"""
        wa, wb = self.where(a), self.where(b)
        if wa == wb:
            wa, wb = self.where(a, sides=True), self.where(b, sides=True)
        return wa, wb

    # sentences
    def describe(self, c):
        """Sentence for one change"""
        lab = c.label
        if not any(o.sure for o in (c.before, c.after) if o is not None):
            lab = f'object that looks like {_article(lab)} {lab}'
        if c.kind == 'moved':
            d = np.array(c.displacement)
            dist = float(np.linalg.norm(d[:2]))
            wa, wb = self.from_to(c.before, c.after)
            turned = c.yaw_deg is not None and abs(c.yaw_deg) >= 20
            if c.details.get('in_place') and c.details.get('measured_by') != 'shape':
                # one of the two views of it is partial, so only say what is sure
                new_side = c.after_assess.status == 'new'
                n = len((c.after_assess if new_side else c.before_assess).confirming_views)
                s = (f'The {lab} was pushed to a new spot overlapping where it stood'
                     + (f' and turned about {abs(c.yaw_deg):.0f}°' if turned else '')
                     + f'; it is now {self.where(c.after, sides=True)}. '
                     + (f'The second video sees it in space the first video saw empty from {n} viewpoints.'
                        if new_side else f'The second video sees empty space where it stood from {n} viewpoints.'))
                return s
            if wa == wb and turned:
                s = (f'The {lab} was turned about {abs(c.yaw_deg):.0f}° and shifted {dist:.1f} m; '
                     f'it is still {wa}.')
            else:
                # left/closer only mean something for a move that is short next to its
                # distance from the start; for longer ones the places say it better
                way = ''
                if dist < 0.5 * np.linalg.norm(c.before.center[:2]):
                    way = f' ({direction_phrase(*in_view(d, c.before.center))})'
                s = f'The {lab} was moved {dist:.1f} m{way}: it was {wa} and is now {wb}.'
                if turned:
                    s += f' It was also turned about {abs(c.yaw_deg):.0f}°.'
            # heights only when both resting places are known (a partly seen object
            # has no reliable bottom)
            if abs(d[2]) > 0.2 and self.support(c.before) and self.support(c.after):
                s += f" It is now {abs(d[2]):.1f} m {'higher' if d[2] > 0 else 'lower'}."
            ba, aa = c.before_assess, c.after_assess
            if ba.reason == 'not_redetected' or (aa is not None and aa.reason == 'not_redetected'):
                s += (' Depth sees straight through it, so the detector checked instead: the second video '
                      f'looked at the old spot from {len(ba.confirming_views)} viewpoints and never found it there')
                if aa is not None and aa.reason == 'not_redetected':
                    s += (f', and the first video looked at the new spot from {len(aa.confirming_views)} '
                          'viewpoints and never found it there either')
                s += '.'
            elif ba.confirming_views:
                s += f' ({len(ba.confirming_views)} views of the second recording see empty space at its old spot.)'
            elif aa is not None and aa.confirming_views:
                s += f' ({len(aa.confirming_views)} views of the first recording saw empty space at its new spot.)'
            return s
        if c.kind == 'likely_moved':
            d = np.array(c.displacement)
            dist = float(np.linalg.norm(d[:2]))
            wa, wb = self.from_to(c.before, c.after)
            gone_a = c.before_assess and c.before_assess.reason == 'not_redetected'
            new_b = c.after_assess and c.after_assess.reason == 'not_redetected'
            if gone_a or new_b:
                n = len((c.before_assess if gone_a else c.after_assess).confirming_views)
                which = 'second' if gone_a else 'first'
                spot = 'old' if gone_a else 'new'
                old, new = wa.replace('on the ', 'the ', 1), wb.replace('on the ', 'the ', 1)
                return (f'The {lab} moved from {old} to {new} '
                        f'({dist:.1f} m). Depth sees straight through it, so the detector checked instead: '
                        f'the {which} video looked at its {spot} spot from {n} viewpoints and never '
                        f'found it there, and a matching {lab} shows up {wb} in a spot the other video never saw '
                        f'clearly. Most likely the same {lab}, moved.')
            return (f'The {lab} that was {wa} is gone from there, and a matching {lab} '
                    f'shows up {wb}, {dist:.1f} m away. That spot was '
                    f'not visible in one of the recordings, so this is most likely the same {lab}, moved.')
        if c.kind == 'added':
            o, a = c.after, c.after_assess
            name = f'{_article(lab)} {lab}' if lab != 'unidentified object' else 'an unidentified object'
            s = f'New: {name} (about {o.size_text()}) appeared {self.where(o)}.'
            if a.reason == 'not_redetected':
                s += (f' Depth sees straight through it, so the detector checked instead: the first video '
                      f'looked at that spot from {len(a.confirming_views)} viewpoints and never found it.')
            elif a.confirming_views:
                s += f' The first recording saw empty space there from {len(a.confirming_views)} viewpoints.'
            return s[0].upper() + s[1:]
        if c.kind == 'removed':
            o, a = c.before, c.before_assess
            name = f'The {lab}' if lab != 'unidentified object' else 'An unidentified object'
            s = f'{name} (about {o.size_text()}) that was {self.where(o)} has been removed.'
            if a.reason == 'not_redetected':
                s += (f' Depth sees straight through it, so the detector checked instead: the second video '
                      f'looked at that spot from {len(a.confirming_views)} viewpoints and never found it.')
            elif a.confirming_views:
                s += (f' The second recording looked at that spot from {len(a.confirming_views)} '
                      f'viewpoints and saw straight through to the surface behind.')
            return s
        if c.kind == 'appearance':
            o = c.before
            d = c.details
            P = o.points - o.points.mean(0)
            normal = np.linalg.svd(P, full_matrices=False)[2][2] if len(P) >= 3 else np.array([0, 0, 1.0])
            if abs(normal[2]) < 0.5:
                place = f'on a wall or other vertical surface, {self.near(o)}'
            else:
                place = self.where(o)
            size = o.size_text().rsplit(' x ', 1)[0]          # flat: width x depth only
            guess = c.details.get('appearance_guess')
            what = f'Something flat ({guess})' if guess else 'Something flat (a rug, mat, papers...)'
            verb = {'removed': 'was removed from', 'added': 'was put down at',
                    'changed': 'changed at'}[d.get('direction', 'changed')]
            place = place.replace('on the ', 'the ', 1) if place.startswith('on the ') else place
            s = (f'{what} {verb} {place} (about {size} cm). The 3D shape there is unchanged, '
                 f'so only the colour gives it away')
            if d.get('direction') == 'removed':
                s += ': the second video now sees the same colour as the surroundings'
            elif d.get('direction') == 'added':
                s += ': it stands out from its surroundings only in the second video'
            return s + '.'
        return ''

    def describe_unverified(self, o, a):
        """Sentence for an object that could not be checked"""
        which = 'first' if o.session == 0 else 'second'
        other = 'second' if o.session == 0 else 'first'
        why = ('it was never in view during the ' + other + ' recording') if a.reason == 'outside_view' \
            else ('other things blocked the view of it in every frame of the ' + other + ' recording')
        name = o.label if o.sure else f'object that looks like {_article(o.label)} {o.label}'
        return (f"Can't tell: the {name} {self.where(o)} (seen in the {which} recording) is not "
                f'reported as changed, because {why}.')

    def headline(self):
        """One line summary of the run"""
        cs = self.cs
        kinds = [c.kind for c in cs.changes]
        parts = []
        for k, word in (('moved', 'moved'), ('likely_moved', 'probably moved'),
                        ('added', 'added'), ('removed', 'removed'), ('appearance', 'flat / colour-only')):
            n = kinds.count(k)
            if n:
                parts.append(f'{n} {word}')
        if not parts:
            head = 'No meaningful changes detected.'
        else:
            head = f"{len(cs.changes)} change{'s' if len(cs.changes) != 1 else ''} found: " + ', '.join(parts) + '.'
        tail = f' {len(cs.unchanged)} objects unchanged'
        if cs.unverified:
            tail += f', {len(cs.unverified)} could not be checked (not visible in both recordings)'
        return head + tail + '.'


def describe_all(cs):
    """Fills in the sentence of every change"""
    d = Describer(cs)
    for c in cs.changes:
        c.sentence = d.describe(c)
    return d
