"""
config.py - all the parameters in one place, in metres after the scale is set
"""

from dataclasses import dataclass, field

# open-vocabulary prompt list for the detector. Structural classes (wall, floor,
# door, window, rug...) are left out on purpose: they don't move, and flat things
# like rugs are invisible to a geometric free-space test anyway
DEFAULT_CLASSES = [
    'chair', 'office chair', 'stool', 'table', 'desk', 'sofa', 'armchair', 'bed',
    'cabinet', 'shelf', 'bookshelf', 'drawer unit', 'lamp', 'plant', 'potted plant',
    'box', 'cardboard box', 'bag', 'backpack', 'suitcase', 'basket', 'bin', 'trash can',
    'laptop', 'monitor', 'tv', 'keyboard', 'speaker', 'fan', 'printer',
    'bottle', 'cup', 'mug', 'bowl', 'vase', 'book', 'pillow', 'cushion', 'blanket',
    'jacket', 'towel', 'shoe', 'ball', 'toy', 'guitar', 'umbrella', 'bucket',
    'clothes rack', 'ironing board', 'vacuum cleaner', 'laundry basket', 'water bottle',
]

# a second pass for everyday things the main list misses (kitchen appliances, small
# electronics...). Grounding DINO splits its confidence between similar words in one
# prompt, so these run in their own prompts, similar words kept apart, and they only
# fill gaps: where the main list already found something, its answer stands
EXTRA_CLASSES = [
    'washing machine', 'microwave', 'kettle', 'computer mouse', 'remote control', 'phone',
    'headphones', 'candle', 'clock', 'picture frame', 'tissue box', 'coffee table',
    'oven', 'air fryer', 'toaster', 'milk carton', 'jar', 'plate', 'game controller',
    'heater', 'router', 'hat', 'dumbbell', 'side table',
    'tv stand', 'kitchen cabinet', 'neck pillow', 'ottoman', 'paper bag', 'notebook',
]

# detector phrases mapped onto one canonical name, so "office chair" and
# "chair" seen in different frames vote for the same object
CANONICAL = {
    'office chair': 'chair', 'armchair': 'chair', 'potted plant': 'plant',
    'cardboard box': 'box', 'trash can': 'bin', 'mug': 'cup', 'cushion': 'pillow',
    'laundry basket': 'basket', 'water bottle': 'bottle', 'drawer unit': 'cabinet',
    'kitchen cabinet': 'cabinet', 'bookshelf': 'shelf', 'coffee table': 'table',
    'side table': 'table', 'computer mouse': 'mouse', 'remote control': 'remote',
    'paper bag': 'bag', 'notebook': 'book',
    'tv': 'TV',
}

# longest side a thing of this kind can plausibly have (m): when the detector is
# torn between two names, the one that fits the measured size wins (a 1.6 m "chair"
# is a sofa)
MAX_SIZE = {
    'chair': 1.1, 'stool': 0.7, 'bottle': 0.45, 'cup': 0.25, 'bowl': 0.4, 'vase': 0.6,
    'book': 0.5, 'bag': 1.0, 'pillow': 1.0, 'speaker': 0.8, 'keyboard': 0.6, 'laptop': 0.6,
    'monitor': 1.4, 'printer': 0.8, 'bin': 0.7, 'lamp': 0.8, 'plant': 1.5, 'shoe': 0.4,
    'ball': 0.5, 'toy': 0.8, 'fan': 0.8, 'box': 1.5, 'mouse': 0.2, 'remote': 0.35,
    'phone': 0.25, 'headphones': 0.3, 'milk carton': 0.35, 'jar': 0.35, 'kettle': 0.4,
    'toaster': 0.5, 'air fryer': 0.55, 'microwave': 0.8, 'plate': 0.4, 'candle': 0.3,
    'clock': 0.6, 'picture frame': 1.2, 'tissue box': 0.35, 'hat': 0.5, 'dumbbell': 0.5,
    'game controller': 0.25, 'router': 0.5, 'neck pillow': 0.45,
}

# shortest longest-side a whole one can have (m): a smaller "chair" is a glimpse of
# something else, so its name is only offered as a guess
MIN_SIZE = {'chair': 0.28, 'sofa': 0.8, 'table': 0.5, 'desk': 0.5, 'bed': 1.0, 'cabinet': 0.3,
            'TV': 0.5, 'washing machine': 0.4, 'oven': 0.35}

# long thin things: a "keyboard" as square as a laptop is the laptop
MIN_ASPECT = {'keyboard': 2.0}

# summed detector score a name needs before the report states it plainly (about three
# confident detections); weaker names are said as "what looks like a ...". The object
# itself is still checked for changes either way
SURE_NAME_VOTES = 1.0

# large furniture is preferred as a landmark when describing where things are
LANDMARK_CLASSES = {'sofa', 'table', 'desk', 'bed', 'cabinet', 'shelf', 'chair',
                    'monitor', 'lamp', 'plant', 'box', 'stool'}


@dataclass
class Config:
    """Tunable parameters"""
    # models (Hugging Face ids)
    vggt_model: str = 'facebook/VGGT-1B'
    detector_model: str = 'IDEA-Research/grounding-dino-base'
    sam_model: str = 'facebook/sam-vit-base'
    clip_model: str = 'openai/clip-vit-base-patch32'
    classes_per_prompt: int = 12      # Grounding DINO works better with short prompts

    # frames / reconstruction
    frames_per_video: int = 24        # keyframes per recording fed jointly to VGGT
    sample_fps: float = 4.0           # candidate frames per second before sharpness selection
    hires_max_side: int = 1024        # frames kept for detection & evidence images
    conf_drop_percentile: float = 30  # drop the least-confident VGGT depth pixels
    edge_rel_thresh: float = 0.06     # drop "flying pixels" at depth discontinuities

    # metric scale & alignment
    camera_height_m: float = 1.40     # typical phone height when filming a room
    refine_sessions: bool = True      # trimmed Sim(3) ICP of the 'after' frames onto 'before'

    # point clouds
    voxel: float = 0.02               # m, scene point-cloud resolution
    object_voxel: float = 0.04        # m, voxel size used to fuse detections into objects

    # visibility test (the heart of the method)
    tau_abs: float = 0.06             # m, depth agreement tolerance ...
    tau_rel: float = 0.04             # ... plus this fraction of the viewing distance
    auto_calibrate: bool = True       # set tau_rel from the measured noise of this run
    tau_sigmas: float = 3.5
    tau_rel_min: float = 0.03
    tau_rel_max: float = 0.10
    min_free_views: int = 2           # independent views that must see "through" a point
    changed_ratio: float = 0.66       # free / (free + same) to call a point changed
    kept_ratio: float = 0.34          # ... and below this it is unchanged

    # objects
    classes: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    extra_classes: list[str] = field(default_factory=lambda: list(EXTRA_CLASSES))
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    min_views_per_object: int = 2
    min_object_points: int = 40
    max_object_extent: float = 3.5    # m; bigger "objects" are usually walls/floor
    object_gone_frac: float = 0.50    # fraction of evaluated points seen-through -> gone/new
    object_maybe_frac: float = 0.30
    min_evaluated_frac: float = 0.15  # below this an object is "unverified"
    moved_min_dist: float = 0.15      # m
    moved_min_yaw: float = 20.0       # deg
    shape_min_fit: float = 0.35       # share of the shape that must land on the other video's surface

    # detector second opinion for see-through / thin objects (recheck.py)
    recheck: bool = True
    recheck_max_size: float = 1.0     # m; bigger things are judged by geometry alone
    recheck_min_looks: int = 4        # own-video views needed to trust the detector ...
    recheck_own_rate: float = 0.75    # ... and how often it must find the object in them
    recheck_min_other_looks: int = 3  # clear looks of the other video that all find nothing

    # unlabelled geometric changes
    region_voxel: float = 0.05
    region_min_points: int = 120
    region_min_extent: float = 0.10   # m
    region_min_height: float = 0.03   # m above floor (ignore floor noise)
    region_min_thickness: float = 0.06  # m; thinner unnamed changes are edge artefacts
    region_max_elongation: float = 6.0  # longest / shortest side of an unnamed change

    # appearance changes on geometrically unchanged surfaces (rugs, papers, posters)
    appearance: bool = True
    app_voxel: float = 0.05
    app_luma_weight: float = 0.3      # brightness counts less than colour (shadows / lighting)
    app_thresh: float = 18.0          # corrected Lab distance (OpenCV 8-bit Lab units)
    app_min_voxels: int = 16          # ~0.04 m^2: about an A4 sheet
    app_min_width: float = 0.15       # m; narrower clusters are boundary slivers
    app_shadow_guard: float = 0.35    # m around geometric changes where colour changes are ignored

    # occupancy map
    grid_res: float = 0.05
    robot_band: tuple[float, float] = (0.08, 1.80)   # m above floor that blocks a robot

    seed: int = 0
