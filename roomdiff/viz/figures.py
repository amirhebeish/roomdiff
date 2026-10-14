"""
figures.py - floor plan, occupancy maps, evidence images and expected vs actual frames
"""


import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Polygon  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from ..objects import main_copy, pair_sure  # noqa: E402
from ..occupancy import (C_CLEARED, C_FREE, C_NEW_OBSTACLE, C_ONE_SIDED, C_STATIC,  # noqa: E402
                         C_UNKNOWN)
from ..scene import AFTER, BEFORE  # noqa: E402
from ..visibility import FREE, SAME, DepthCache, ray_check  # noqa: E402

# one palette everywhere: map, evidence images, 3D viewer, report
COLORS = {
    'moved': '#3E63DD', 'likely_moved': '#3E63DD', 'added': '#30A46C',
    'removed': '#E5484D', 'appearance': '#8E4EC6', 'unverified': '#F5A524', 'static': '#5B5B5B',
    'before_cam': '#9A9A9A', 'after_cam': '#2B2B2B',
}
KIND_TEXT = {'moved': 'MOVED', 'likely_moved': 'PROBABLY MOVED', 'added': 'ADDED',
             'removed': 'REMOVED', 'appearance': 'LOOKS DIFFERENT', 'unverified': "CAN'T TELL"}


def _hex(c):
    c = c.lstrip('#')
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _font(size):
    for name in ('DejaVuSans-Bold.ttf', 'Arial Bold.ttf', 'arialbd.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _footprint_poly(o):
    """Corners of the footprint rectangle"""
    cx, cy, w, d, ang = o.footprint_box()
    return cv2.boxPoints(((cx, cy), (w, d), ang))


def overview_map(res, path):
    """Top down floor plan with the changes and both camera paths"""
    occ, cs = res.occ, res.cs
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=130)
    ext = occ.extent()
    base = np.full(occ.change.shape + (3,), 1.0)
    base[occ.change == C_UNKNOWN] = (0.93, 0.93, 0.93)
    base[occ.change == C_ONE_SIDED] = (0.97, 0.97, 0.97)
    both = (occ.before == 2) & (occ.after == 2)
    base[np.isin(occ.change, (C_STATIC, C_CLEARED, C_NEW_OBSTACLE)) & both] = (0.55, 0.55, 0.55)
    base[occ.change == C_STATIC] = (0.55, 0.55, 0.55)
    ax.imshow(base, origin='lower', extent=ext, interpolation='nearest')

    placed = []

    def free_spot(c, text):
        # nudge a label down until it no longer sits on one already placed
        x, y = float(c[0]), float(c[1])
        w = 0.07 * len(text)
        for _ in range(8):
            if all(abs(x - px) > 0.5 * (w + pw) or abs(y - py) > 0.17 for px, py, pw in placed):
                break
            y -= 0.17
        placed.append((x, y, w))
        return x, y

    for o, _ in cs.unverified:
        ax.add_patch(Polygon(_footprint_poly(o), closed=True, fill=True, fc=COLORS['unverified'] + '33',
                             ec=COLORS['unverified'], lw=1.4, ls='--'))
        x, y = free_spot(o.center, f'? {o.label}')
        ax.text(x, y, f'? {o.label}', fontsize=8, color='#8a5a00', ha='center', va='center', weight='bold')

    for a, b in cs.unchanged:
        o = a if a is not None else b
        ax.add_patch(Polygon(_footprint_poly(o), closed=True, fill=False, ec=COLORS['static'], lw=0.8, ls=':'))
        if pair_sure(a, b):
            name = main_copy(a, b).label
            x, y = free_spot(o.center, name)
            ax.text(x, y, name, fontsize=7, color='#444', ha='center', va='center')

    for ch in cs.changes:
        col = COLORS[ch.kind]
        if ch.before is not None:
            ax.add_patch(Polygon(_footprint_poly(ch.before), closed=True, fill=ch.kind == 'removed',
                                 fc=col + '40', ec=col, lw=1.8, ls='--' if ch.kind != 'removed' else '-'))
        if ch.after is not None:
            ax.add_patch(Polygon(_footprint_poly(ch.after), closed=True, fill=True, fc=col + '55', ec=col, lw=2))
        if ch.before is not None and ch.after is not None:
            p0, p1 = ch.before.center[:2], ch.after.center[:2]
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=16, color=col, lw=2,
                                         shrinkA=6, shrinkB=6))
        anchor = (ch.after or ch.before).center
        ax.annotate(f'{ch.id} {ch.label}', anchor[:2], xytext=(8, 8), textcoords='offset points',
                    fontsize=9, weight='bold', color=col,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=col, lw=0.8, alpha=0.9))

    for s, key, mk in ((BEFORE, 'before_cam', 'o'), (AFTER, 'after_cam', 's')):
        C = res.scene.cam_centers(res.scene.frames_of(s))
        ax.plot(C[:, 0], C[:, 1], '-', color=COLORS[key], lw=1, alpha=0.8)
        ax.plot(C[:, 0], C[:, 1], mk, color=COLORS[key], ms=3)
        # viewing directions
        for i in res.scene.frames_of(s)[::3]:
            f = res.scene.T_cw[i, 2, :3]
            c = res.scene.cam_centers([i])[0]
            ax.plot([c[0], c[0] + 0.3 * f[0]], [c[1], c[1] + 0.3 * f[1]], color=COLORS[key], lw=0.7)
    ax.annotate('start', (0, 0), xytext=(0.45, 0), arrowprops=dict(arrowstyle='<-', color='#666'),
                fontsize=7, color='#666', va='center')

    handles = [Patch(fc=COLORS['moved'] + '55', ec=COLORS['moved'], label='moved (dashed = old place)'),
               Patch(fc=COLORS['added'] + '55', ec=COLORS['added'], label='added'),
               Patch(fc=COLORS['removed'] + '40', ec=COLORS['removed'], label='removed'),
               Patch(fc=COLORS['unverified'] + '33', ec=COLORS['unverified'], ls='--',
                     label="can't tell (not seen twice)"),
               Patch(fc=(0.55, 0.55, 0.55), label='unchanged obstacles'),
               Line2D([], [], color=COLORS['before_cam'], marker='o', ms=3, label='camera path, before'),
               Line2D([], [], color=COLORS['after_cam'], marker='s', ms=3, label='camera path, after')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_xlabel('x (m) — away from the starting viewpoint')
    ax.set_ylabel('y (m) — to the left')
    ax.set_aspect('equal')
    ax.set_title(res.describer.headline(), fontsize=9, loc='left', wrap=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def occupancy_figure(res, path):
    """Before, after and change occupancy maps"""
    occ = res.occ
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=120)
    lut = np.array([[0.85, 0.85, 0.85], [1, 1, 1], [0.15, 0.15, 0.15]])
    for ax, grid, title in ((axes[0], occ.before, 'Before'), (axes[1], occ.after, 'After')):
        ax.imshow(lut[grid], origin='lower', extent=occ.extent(), interpolation='nearest')
        ax.set_title(f"{title}: occupancy at robot height ({res.info.get('band', '0.08–1.8 m')})", fontsize=9)
    cl = np.zeros(occ.change.shape + (3,))
    cl[occ.change == C_UNKNOWN] = (0.85, 0.85, 0.85)
    cl[occ.change == C_ONE_SIDED] = (0.93, 0.93, 0.93)
    cl[occ.change == C_FREE] = (1, 1, 1)
    cl[occ.change == C_STATIC] = (0.35, 0.35, 0.35)
    cl[occ.change == C_NEW_OBSTACLE] = np.array(_hex(COLORS['added'])) / 255
    cl[occ.change == C_CLEARED] = np.array(_hex(COLORS['removed'])) / 255
    axes[2].imshow(cl, origin='lower', extent=occ.extent(), interpolation='nearest')
    axes[2].set_title('Change for navigation', fontsize=9)
    axes[2].legend(handles=[Patch(fc=COLORS['added'], label='newly blocked'),
                            Patch(fc=COLORS['removed'], label='newly free'),
                            Patch(fc=(0.35, 0.35, 0.35), label='unchanged obstacle'),
                            Patch(fc=(0.93, 0.93, 0.93), ec='#bbb', label='seen in one recording only'),
                            Patch(fc=(0.85, 0.85, 0.85), label='never seen')],
                   fontsize=7, loc='lower right', framealpha=0.9)
    for ax in axes:
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def _load_frame(scene, i, scale_to=640):
    """Best image of frame i and the scale from depth pixels to image pixels"""
    if scene.hires_paths:
        try:
            im = Image.open(scene.hires_paths[i]).convert('RGB')
            return im, im.width / scene.hw[1]
        except Exception:
            pass
    im = Image.fromarray(scene.images[i])
    f = scale_to / im.width
    return im.resize((scale_to, int(im.height * f)), Image.BICUBIC), f


def _best_frame(points, scene, frames, want, cache):
    """Frame where most of the points get the wanted evidence code"""
    if points is None or len(frames) == 0:
        return None, None
    sub = points if len(points) < 4000 else points[np.random.default_rng(0).choice(len(points), 4000, replace=False)]
    pv = ray_check(sub, scene, frames, cache=cache)
    cnt = (pv == want).sum(1)
    # prefer views where the object is near the image centre (less cropped)
    k = int(np.argmax(cnt))
    if cnt[k] < 5:
        return None, None
    return int(frames[k]), sub[pv[k] == want] if want == SAME else sub[pv[k] != 0]


def _bbox(points, scene, i, f):
    """Image box of the projected points"""
    u, v, z = scene.project(points, i)
    ok = z > 0
    if ok.sum() < 3:
        return None
    u, v = u[ok], v[ok]
    H, W = scene.hw
    box = [np.percentile(u, 2), np.percentile(v, 2), np.percentile(u, 98), np.percentile(v, 98)]
    box = [np.clip(box[0], 0, W - 1), np.clip(box[1], 0, H - 1), np.clip(box[2], 0, W - 1), np.clip(box[3], 0, H - 1)]
    return [b * f for b in box]


def _dashed_rect(draw, box, color, width=3, dash=10):
    x0, y0, x1, y1 = box
    for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        L = np.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(L // dash))
        for k in range(0, n, 2):
            t0, t1 = k / n, min(1, (k + 1) / n)
            draw.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                       (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)], fill=color, width=width)


def _tag(draw, xy, text, color, font):
    x, y = xy
    tb = draw.textbbox((0, 0), text, font=font)
    w, h = tb[2] - tb[0], tb[3] - tb[1]
    y = max(0, y - h - 8)
    draw.rectangle([x, y, x + w + 10, y + h + 8], fill=color)
    draw.text((x + 5, y + 3), text, fill='white', font=font)


def _hatch(rgb, mask, step=8):
    """Hatches the pixels the memory never saw"""
    out = rgb.copy()
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    stripe = ((xx + yy) // max(2, step // 2)) % 2 == 0
    out[mask] = np.where(stripe[mask, None], (58, 60, 66), (34, 36, 40))
    return out


def expected_image(res, mr, frame):
    """Expected view of a frame, hatched where never seen"""
    rgb, depth, src = mr.render(frame)
    return _hatch(rgb, ~np.isfinite(depth)), depth, src


def _panel(scene, frame, solid_pts, dashed_pts, color, solid_txt, dashed_txt, title, image=None):
    """One captioned image with the object boxes drawn"""
    if image is None:
        im, f = _load_frame(scene, frame)
    else:                                   # an array at depth resolution (e.g. a rendered expected view)
        base = Image.fromarray(image)
        f = 640 / base.width
        im = base.resize((640, int(base.height * f)), Image.BICUBIC)
        hi, fh = _load_frame(scene, frame)
        if hi.width != im.width:            # match the photo panels' size
            g = hi.width / im.width
            im = im.resize((hi.width, int(im.height * g)), Image.BICUBIC)
            f *= g
    im = im.copy()
    d = ImageDraw.Draw(im)
    font = _font(max(14, im.width // 40))
    lw = max(3, im.width // 220)
    if dashed_pts is not None:
        b = _bbox(dashed_pts, scene, frame, f)
        if b:
            _dashed_rect(d, b, color, width=lw)
            _tag(d, (b[0], b[1]), dashed_txt, color, font)
    if solid_pts is not None:
        b = _bbox(solid_pts, scene, frame, f)
        if b:
            d.rectangle(b, outline=color, width=lw)
            _tag(d, (b[0], b[1]), solid_txt, color, font)
    bar = Image.new('RGB', (im.width, max(30, im.width // 22)), (24, 24, 24))
    bd = ImageDraw.Draw(bar)
    bd.text((10, 6), title, fill='white', font=_font(max(13, im.width // 48)))
    out = Image.new('RGB', (im.width, im.height + bar.height))
    out.paste(im, (0, 0))
    out.paste(bar, (0, im.height))
    return out


def _in_view(points, scene, i, frac=0.6):
    if points is None:
        return False
    u, v, z = scene.project(points, i)
    H, W = scene.hw
    return ((z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).mean() > frac


def _compose(panels, header, col, path):
    """Panels side by side under a coloured header"""
    h = max(p.height for p in panels)
    panels = [p.resize((int(p.width * h / p.height), h)) for p in panels]
    gap = 8
    W = sum(p.width for p in panels) + gap * (len(panels) - 1)
    head_h = max(36, h // 12)
    out = Image.new('RGB', (W, h + head_h), (255, 255, 255))
    hd = ImageDraw.Draw(out)
    hd.rectangle([0, 0, W, head_h], fill=col)
    hd.text((12, head_h // 2 - head_h // 4), header, fill='white', font=_font(max(15, head_h // 2)))
    x = 0
    for p in panels:
        out.paste(p, (x, head_h))
        x += p.width + gap
    out.save(path, quality=88)
    return path


def evidence_image(res, change, path, cache=None, mr=None):
    """Before photo | expected from the after camera | actual after photo"""
    scene = res.scene
    cache = cache or DepthCache(scene)
    fA, fB = scene.frames_of(BEFORE), scene.frames_of(AFTER)
    col = _hex(COLORS[change.kind])
    lab = change.label
    b, a = change.before, change.after
    flat = change.kind == 'appearance'
    panels = []

    # 1. the original photo from the first video
    if b is not None:
        i, vis = _best_frame(b.points, scene, fA, SAME, cache)
        if i is not None:
            panels.append(_panel(scene, i, b.points if flat else vis, None, col, lab, '',
                                 f'BEFORE · video 1 · {scene.frame_names[i]}'))
    else:
        i, _ = _best_frame(a.points, scene, fA, FREE, cache)
        if i is not None:
            panels.append(_panel(scene, i, None, a.points, col, '', 'empty here',
                                 f'BEFORE · video 1 · {scene.frame_names[i]}'))

    # 2 + 3. one camera of the second video: what memory predicts vs what it sees
    if a is not None:
        j, _ = _best_frame(a.points, scene, fB, SAME, cache)
    else:
        j, _ = _best_frame(b.points, scene, fB, FREE, cache)
    if j is not None:
        b_here = b if (b is not None and _in_view(b.points, scene, j)) else None
        a_here = a if (a is not None and _in_view(a.points, scene, j, 0.3)) else None
        if mr is not None:
            exp, _, _ = expected_image(res, mr, j)
            if flat:
                panels.append(_panel(scene, j, b.points, None, col, 'expected', '',
                                     'EXPECTED from this camera · memory of video 1', image=exp))
            else:
                panels.append(_panel(scene, j, b_here.points if b_here is not None else None,
                                     a_here.points if a_here is not None else None, col,
                                     f'{lab} (expected)', 'empty in memory',
                                     'EXPECTED from this camera · memory of video 1', image=exp))
        if flat:
            panels.append(_panel(scene, j, a.points, None, col, 'now', '',
                                 f'ACTUAL · video 2 · {scene.frame_names[j]}'))
        else:
            panels.append(_panel(scene, j, a_here.points if a_here is not None else None,
                                 b_here.points if b_here is not None else None, col,
                                 f'{lab} (now)', 'gone' if a is None else 'was here',
                                 f'ACTUAL · video 2 · {scene.frame_names[j]}'))
    if not panels:
        return None
    return _compose(panels, f'{change.id}  {KIND_TEXT[change.kind]}: {lab}', col, path)


def unverified_image(res, obj, path, cache=None):
    """Photo of an object that could not be checked"""
    scene = res.scene
    cache = cache or DepthCache(scene)
    own = scene.frames_of(obj.session)
    i, vis = _best_frame(obj.points, scene, own, SAME, cache)
    if i is None:
        return None
    p = _panel(scene, i, vis, None, _hex(COLORS['unverified']), f'? {obj.label}', '',
               f"{'BEFORE · video 1' if obj.session == BEFORE else 'AFTER · video 2'} · {scene.frame_names[i]}")
    p.save(path, quality=85)
    return path


def surprise_frames(res, mr, out_dir, cfg, max_frames=24, panel_w=420):
    """Expected | actual | surprise for every after camera, plus a gif"""
    from ..expect import verdict_lookup
    from ..scene import encode_voxels
    from ..visibility import CHANGED
    scene = res.scene
    lookA = verdict_lookup(res.ps[BEFORE], res.ev[BEFORE], cfg.voxel)
    lookB = verdict_lookup(res.ps[AFTER], res.ev[AFTER], cfg.voxel)
    app_keys = np.unique(np.concatenate(
        [encode_voxels(c.after.points, cfg.app_voxel) for c in res.cs.changes if c.kind == 'appearance']
        or [np.zeros(0, np.int64)]))
    frames = scene.frames_of(AFTER)
    if len(frames) > max_frames:
        frames = frames[np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
    H, W = scene.hw
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, gif, scores = [], [], []
    red, green, purple = (np.array(_hex(COLORS[k]), float) for k in ('removed', 'added', 'appearance'))
    for n, j in enumerate(frames):
        exp_rgb, exp_d, src = mr.render(j)
        never = ~np.isfinite(exp_d)
        gone = np.zeros((H, W), bool)
        has = src >= 0
        gone[has] = lookA(mr.P[src[has]]) == CHANGED
        pts, _, pix = scene.backproject(j)
        new = np.zeros((H, W), bool)
        app = np.zeros((H, W), bool)
        vb = lookB(pts)
        new[pix[:, 0], pix[:, 1]] = vb == CHANGED
        if len(app_keys):
            app[pix[:, 0], pix[:, 1]] = np.isin(encode_voxels(pts, cfg.app_voxel), app_keys)
        act = scene.images[j].astype(float)
        grey = act.mean(2, keepdims=True) * 0.55 + 20
        sur = np.repeat(grey, 3, 2)
        sur[gone] = 0.35 * sur[gone] + 0.65 * (0.45 * exp_rgb[gone] + 0.55 * red)
        sur[new] = 0.3 * sur[new] + 0.7 * green
        sur[app & ~new] = 0.35 * sur[app & ~new] + 0.65 * purple
        sur = _hatch(sur.clip(0, 255).astype(np.uint8), never & ~new & ~gone)
        tiles = [(_hatch(exp_rgb, never), 'EXPECTED  (memory of video 1)'),
                 (scene.images[j], f'ACTUAL  (video 2, {scene.frame_names[j]})'),
                 (sur, 'SURPRISE  red gone · green new · purple recoloured')]
        row = []
        for arr, title in tiles:
            im = Image.fromarray(arr).resize((panel_w, int(H * panel_w / W)), Image.BICUBIC)
            bar = Image.new('RGB', (panel_w, 26), (20, 21, 24))
            ImageDraw.Draw(bar).text((8, 5), title, fill=(235, 235, 235), font=_font(13))
            tile = Image.new('RGB', (panel_w, im.height + 26))
            tile.paste(im, (0, 0))
            tile.paste(bar, (0, im.height))
            row.append(tile)
        frame_img = Image.new('RGB', (panel_w * 3 + 12, row[0].height), (255, 255, 255))
        for k, t in enumerate(row):
            frame_img.paste(t, (k * (panel_w + 6), 0))
        scores.append(int(gone.sum() + new.sum() + app.sum()))
        p = out_dir / f'surprise_{n:02d}.jpg'
        frame_img.save(p, quality=82)
        paths.append(p)
        gif.append(frame_img.convert('P', palette=Image.ADAPTIVE, colors=192))
    if gif:
        gif[0].save(out_dir.parent / 'expected_vs_actual.gif', save_all=True, append_images=gif[1:],
                    duration=700, loop=0, optimize=True)
        # a single still of the most surprising view, for the README
        paths[int(np.argmax(scores))].with_name('most_surprising.jpg').write_bytes(
            paths[int(np.argmax(scores))].read_bytes())
    return paths, (int(np.argmax(scores)) if scores else 0)


def inventory_items(res):
    """Everything the run found, one entry per physical object: (object, status, colour)"""
    cs = res.cs
    items = []
    for c in cs.changes:
        if c.kind == 'appearance':           # one flat patch, seen twice
            items.append((c.before, f'{c.id} {KIND_TEXT[c.kind].lower()}', COLORS[c.kind]))
            continue
        for o, when in ((c.before, 'before'), (c.after, 'after')):
            if o is not None:
                items.append((o, f'{c.id} {KIND_TEXT[c.kind].lower()} ({when})', COLORS[c.kind]))
    for a, b in cs.unchanged:
        o = main_copy(a, b)
        status = 'unchanged' if o.sure else 'unchanged, name unsure'
        if a is None or b is None:
            status += f' (video {o.session + 1} only)'
        items.append((o, status, COLORS['static']))
    for o, _ in cs.unverified:
        items.append((o, "can't tell", COLORS['unverified']))
    return items


def inventory_sheet(res, path, cache=None, tile=260, per_row=6):
    """One cropped photo per object found, numbered as in the report"""
    scene = res.scene
    cache = cache or DepthCache(scene)
    font, small = _font(15), _font(12)
    tiles = []
    for k, (o, status, col) in enumerate(inventory_items(res)):
        i, vis = _best_frame(o.points, scene, scene.frames_of(o.session), SAME, cache)
        if i is None:
            i, vis = _best_frame(o.points, scene, scene.frames_of(o.session), FREE, cache)
        canvas = Image.new('RGB', (tile, tile + 46), 'white')
        d = ImageDraw.Draw(canvas)
        name = o.label + ('' if o.sure else '?')
        d.text((6, 4), f'#{k + 1} {name}', fill='black', font=font)
        d.text((6, 25), status, fill=col, font=small)
        if i is not None:
            im, f = _load_frame(scene, i)
            box = _bbox(vis, scene, i, f)
            if box is not None:
                dr = ImageDraw.Draw(im)
                dr.rectangle(box, outline=_hex(col), width=max(2, int(3 * f / 2)))
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                half = max(box[2] - box[0], box[3] - box[1], 60 * f) * 0.75
                crop = im.crop((int(max(0, cx - half)), int(max(0, cy - half)),
                                int(min(im.width, cx + half)), int(min(im.height, cy + half))))
                canvas.paste(crop.resize((tile, tile)), (0, 46))
        tiles.append(canvas)
    if not tiles:
        return None
    rows = (len(tiles) + per_row - 1) // per_row
    sheet = Image.new('RGB', (per_row * (tile + 6), rows * (tile + 52)), 'white')
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % per_row) * (tile + 6), (k // per_row) * (tile + 52)))
    sheet.save(path, quality=85)
    return path
