"""
make_diagram.py - draws docs/images/visibility.png, the three outcomes of the per ray test

Run from the repository root: python docs/make_diagram.py
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

C = {'same': '#5B5B5B', 'free': '#E5484D', 'occ': '#F5A524', 'cam': '#2B2B2B'}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
cases = [
    ('SAME: surface still there', 'same', False, 3.0,
     'depth measured ≈ depth of point\n→ evidence the object is still there'),
    ('FREE: seen straight through', 'free', False, 5.0, 'measured depth ≫ point depth\n→ positive evidence it is GONE'),
    ('OCCLUDED: no evidence', 'occ', True, 1.8,
     "something nearer blocks the ray\n→ can't tell (never called 'removed')"),
]
for ax, (title, key, occluder, hit, note) in zip(axes, cases):
    ax.set_xlim(-0.6, 6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(Rectangle((5.0, -1.4), 0.2, 2.8, color='#c9c9c9'))          # back wall
    ax.text(5.1, 1.5, 'wall', ha='center', fontsize=8, color='#777')
    ax.add_patch(Polygon([[0, 0], [-0.45, 0.28], [-0.45, -0.28]], color=C['cam']))   # camera
    ax.text(-0.25, -0.85, "'after'\ncamera", ha='center', fontsize=8)
    # the point from the 'before' recording
    ax.add_patch(Circle((3.0, 0), 0.09, color='#3E63DD', zorder=5))
    ax.text(3.0, 0.85, "point seen\nin 'before'", ha='center', fontsize=8, color='#3E63DD')
    if key == 'same':
        ax.add_patch(Rectangle((3.0, -0.6), 0.5, 1.2, color='#8d8d8d', alpha=0.6))
    if occluder:
        ax.add_patch(Rectangle((1.7, -0.6), 0.25, 1.2, color='#8d8d8d'))
        ax.text(1.82, -0.85, 'occluder', ha='center', fontsize=8, color='#555')
    ax.add_patch(FancyArrowPatch((0.05, 0), (hit, 0), arrowstyle='-|>', mutation_scale=14, color=C[key], lw=2.2))
    ax.plot([hit], [0], 'x', color=C[key], ms=10, mew=2.5)
    ax.set_title(title, fontsize=11, color=C[key] if key != 'same' else '#333', weight='bold')
    ax.text(2.7, -1.35, note, ha='center', fontsize=9, color='#333')
fig.tight_layout()
fig.savefig('docs/images/visibility.png', bbox_inches='tight', facecolor='white')
