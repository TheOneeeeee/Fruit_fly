import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import colorsys
import json
import math
import random
import textwrap
from collections import deque

import numpy as np
import pandas as pd
import pygame

from lif_stepper import LIFBrain
from stimuli import STIMULI

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
FPS = 60
WORLD_W, PANEL_W, H = 750, 1050, 750
LOG_H = 160  # clinical neural event log strip, below the world + brain panels
STEPS_PER_FRAME = 15
FLY_DISPLAY_SIZE = 72
DISPLAY_DECAY = 0.85      # per-frame smoothing for the panel readouts
GLOW_DECAY = 0.985        # per-substep decay of the brain-panel glow
GLOW_THRESHOLD = 0.05     # glow below this is not drawn
TRAIL_HOT_THRESHOLD = 0.55  # glow above this counts as "actively firing" for spike trails
TRAIL_TARGETS_PER_NEURON = 3  # top-K strongest real downstream synapses kept for trail lookup
FEAR_HZ = STIMULI["fear"]["hz"]
# Smell isn't blocked by walls or limited to a forward cone the way sight naturally
# is (compute_vision's radius is gated by the fly's own genome sight_radius) — it
# reaches anywhere in the room. There's no real occlusion geometry in this world
# either way, so "world-spanning" is the honest way to model "unlike sight".
SMELL_RADIUS = math.hypot(WORLD_W, H)

PINK = (255, 105, 180)
TEXT_LIGHT = (205, 205, 215)
EYE_TARGET_COLORS = [(158, 41, 59), (116, 26, 51)]

# --- Brain-panel colors -----------------------------------------------------
# Every neuron in the subgraph structurally belongs to one behavior — the one
# build_subgraph.py found it was most strongly, synaptically driven by (see
# data/subgraph_owner.npy). That ownership is what colors a neuron; how
# brightly it's drawn is separately controlled by how recently it actually
# fired (category_glow). A neuron with no recorded owner (an older subgraph
# built before ownership tracking existed) falls back to the ambient color.
(CAT_AMBIENT, CAT_SUGAR, CAT_BITTER, CAT_WATER, CAT_SALT, CAT_FEAR,
 CAT_AGGRESSION, CAT_REWARD, CAT_PUNISH, CAT_MATING) = range(10)

CATEGORY_COLORS = {
    CAT_AMBIENT:    (150, 160, 190),  # background vision/motion - muted slate
    CAT_SUGAR:      (80, 220, 90),    # green
    CAT_BITTER:     (200, 120, 40),   # burnt orange
    CAT_WATER:      (60, 150, 255),   # blue
    CAT_SALT:       (40, 220, 210),   # teal — near-white was invisible once brightness-scaled dim
    CAT_FEAR:       (255, 60, 60),    # red
    CAT_AGGRESSION: (170, 30, 90),    # deep magenta
    CAT_REWARD:     (255, 210, 60),   # gold
    CAT_PUNISH:     (150, 70, 230),   # purple
    CAT_MATING:     PINK,
}
CATEGORY_PALETTE = np.array([CATEGORY_COLORS[i] for i in range(len(CATEGORY_COLORS))], dtype=np.float32)
LEGEND = [("AMBIENT", CAT_AMBIENT), ("SUGAR", CAT_SUGAR), ("BITTER", CAT_BITTER), ("WATER", CAT_WATER),
          ("SALT", CAT_SALT), ("FEAR", CAT_FEAR), ("AGGR", CAT_AGGRESSION), ("REWARD", CAT_REWARD),
          ("PUNISH", CAT_PUNISH), ("MATING", CAT_MATING)]

CATEGORY_NAME_TO_ID = {
    "sugar": CAT_SUGAR, "bitter": CAT_BITTER, "water": CAT_WATER, "salt": CAT_SALT,
    "fear": CAT_FEAR, "aggression": CAT_AGGRESSION, "dopamine_reward": CAT_REWARD,
    "dopamine_punish": CAT_PUNISH, "mating": CAT_MATING,
    "motion": CAT_AMBIENT, "vision_left": CAT_AMBIENT, "vision_right": CAT_AMBIENT,
    "olfactory_left": CAT_AMBIENT, "olfactory_right": CAT_AMBIENT,
}
CATEGORY_ID_TO_LABEL = {cat: label for label, cat in LEGEND}


def _load_neuron_owners():
    """id -> category for every neuron build_subgraph.py assigned an owner to.
    Missing file (older subgraph) just means everything falls back to ambient."""
    try:
        ids = np.load("data/subgraph_ids.npy")
        owners = np.load("data/subgraph_owner.npy")
    except FileNotFoundError:
        return {}
    return {int(nid): CATEGORY_NAME_TO_ID.get(name, CAT_AMBIENT) for nid, name in zip(ids, owners)}


NEURON_OWNER = _load_neuron_owners()

# A recruited (non-core) neuron only shows its owner's color while that
# behavior is actually live at or above this level — the same level the
# FEAR/MATING readouts use to decide their own text color. Below it, the
# neuron draws as ambient: real activity is still visible, it just isn't
# mislabeled as "fear" or "mating" from incidental background firing.
CATEGORY_ACTIVE_THRESHOLD = 0.03
FEED_CATEGORY = {"sugar": CAT_SUGAR, "water": CAT_WATER, "salt": CAT_SALT}

# --- Feeding / drinking events ---------------------------------------------
FEED_EVENT_DURATION_MS = 2000
FEED_PICKUP_DIST = 18
FEED_EVENT_COLORS = {
    "sugar": (70, 255, 100),
    "water": (70, 175, 255),
    "salt": (255, 245, 120),
}
FEED_EVENT_LABELS = {"sugar": "EATING SUGAR", "water": "DRINKING WATER", "salt": "EATING SALT"}

# --- Mating season ----------------------------------------------------------
MATING_SEASON_MIN_INTERVAL_MS = 5 * 60 * 60 * 1000   # 5 hours
MATING_SEASON_MAX_INTERVAL_MS = 10 * 60 * 60 * 1000  # 10 hours
MATING_ATTRACTION_STRENGTH = 0.045
MATING_ATTRACTION_MAX = 0.12
MATING_HEART_SPAWN_INTERVAL = 18
MATING_COURTING_RADIUS = 140  # mating neurons start firing inside this distance

# Close pairs stop beelining and circle each other instead.
ORBIT_ENTER_DIST = 110
ORBIT_RADIUS = 40
ORBIT_STEER_STRENGTH = 0.09
DANCE_ANGULAR_SPEED = 0.045

# A pair produces offspring after circling at close range long enough.
COURTSHIP_TRIGGER_MIN_DIST = 55
COURTSHIP_TRIGGER_MAX_DIST = 105
COURTSHIP_DURATION_MS = 3500

# --- Hearts -----------------------------------------------------------------
HEART_FULL_GIF = "RED_Heart.gif"
HEART_DAMAGED_GIF = "broken_heart.gif"
HEART_RENDER_SIZE = 34
HEART_FRAME_MS = 100

# --- Predators --------------------------------------------------------------
FROG_DETECTION_RADIUS = 75
FROG_FEAR_RADIUS = 70
FROG_CHASE_DURATION_MS = 5000
FROG_CHASE_COOLDOWN_MS = 3500
FROG_PATROL_SPEED = 0.65
FROG_CHASE_SPEED = 1.05
FROG_JUMP_DISTANCE_MIN = 55
FROG_JUMP_DISTANCE_MAX = 115
FROG_JUMP_PAUSE_MS = 350

# The spider never chases: it only fires a web at a fly that wanders into range.
SPIDER_WEB_RANGE = 66
SPIDER_SHOT_INTERVAL_MS = 2600
SPIDER_ATTACK_FLASH_MS = 450
SPIDER_SPEED = 0.45
SPIDER_TURN_INTERVAL_MS = 1200
WEB_SPEED = 5.0
WEB_STICK_MS = 1800

GENOME_TRAITS = ["eye_hue", "turn_gain", "sight_radius", "base_speed"]

# ----------------------------------------------------------------------------
# pygame setup, fonts
# ----------------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WORLD_W + PANEL_W, H + LOG_H))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 25)
small_font = pygame.font.SysFont(None, 20)
panel_font = pygame.font.SysFont(None, 23)
panel_bold = pygame.font.SysFont(None, 25, bold=True)
event_font = pygame.font.SysFont(None, 42, bold=True)


# ----------------------------------------------------------------------------
# Asset helpers
# ----------------------------------------------------------------------------
def recolor_eyes(frame, hue_shift_degrees):
    """Hue-shift the eye pixels of a fly frame."""
    frame = frame.copy()
    arr = pygame.surfarray.pixels3d(frame)
    rgb = arr.astype(int)
    mask = np.zeros(arr.shape[:2], dtype=bool)
    for er, eg, eb in EYE_TARGET_COLORS:
        mask |= (np.abs(rgb[..., 0] - er) <= 10) & (np.abs(rgb[..., 1] - eg) <= 10) & (np.abs(rgb[..., 2] - eb) <= 10)
    for x, y in np.argwhere(mask):
        hh, ss, vv = colorsys.rgb_to_hsv(*(rgb[x, y] / 255))
        nr, ng, nb = colorsys.hsv_to_rgb((hh + hue_shift_degrees / 360.0) % 1.0, ss, vv)
        arr[x, y] = (int(nr * 255), int(ng * 255), int(nb * 255))
    del arr
    return frame


def load_gif_frames(path, max_size=HEART_RENDER_SIZE):
    """Load an animated gif as pygame surfaces. Returns [] if unavailable."""
    try:
        from PIL import Image, ImageSequence
    except ImportError:
        print("Pillow not installed - heart gifs disabled (pip install pillow)")
        return []
    try:
        img = Image.open(path)
    except (FileNotFoundError, OSError):
        print(f"Heart gif not found: {path}")
        return []
    frames = []
    for frame in ImageSequence.Iterator(img):
        rgba = frame.convert("RGBA")
        scale = min(max_size / rgba.width, max_size / rgba.height)
        size = (max(8, int(rgba.width * scale)), max(8, int(rgba.height * scale)))
        rgba = rgba.resize(size, Image.NEAREST)
        frames.append(pygame.image.fromstring(rgba.tobytes(), rgba.size, "RGBA").convert_alpha())
    return frames


def sheet_frames(sheet, cell, rows, cols, size):
    """Cut non-empty cells out of a sprite sheet and scale them to size x size."""
    frames = []
    for row in rows:
        for col in cols:
            img = sheet.subsurface((col * cell, row * cell, cell, cell))
            if img.get_bounding_rect().width > 0:
                frames.append(pygame.transform.scale(img, (size, size)))
    return frames


def load_image(path):
    return pygame.image.load(path).convert_alpha()


heart_full_frames = load_gif_frames(HEART_FULL_GIF)
heart_damaged_frames = load_gif_frames(HEART_DAMAGED_GIF)
if not heart_full_frames:
    heart_full_frames = heart_damaged_frames
if not heart_damaged_frames:
    heart_damaged_frames = heart_full_frames

# --- Sprites ---
fruit_sheet = load_image("Fruit_.png")
fruit_sprites = []
for _row in range(fruit_sheet.get_height() // 16):
    for _col in range(fruit_sheet.get_width() // 16):
        _cell = fruit_sheet.subsurface((_col * 16, _row * 16, 16, 16))
        if _cell.get_at((8, 8))[3] > 0:
            fruit_sprites.append(pygame.transform.scale(_cell, (32, 32)))

water_sprite = pygame.transform.scale(load_image("water.png"), (28, 28))
salt_sprites = [pygame.transform.scale(load_image(f"salt_{i}.png"), (28, 28)) for i in (1, 2, 3)]

fly_sheet = load_image("Giant_Fly_Sprite_Sheet.png")


def _fly_cell(row, col, size=None):
    img = fly_sheet.subsurface((col * 32, row * 32, 32, 32))
    return pygame.transform.scale(img, (size, size)) if size else img


fly_frames_raw = [_fly_cell(0, c) for c in range(4)]
GROW_FRAMES = [_fly_cell(3, c, FLY_DISPLAY_SIZE) for c in range(3)]      # small -> big
EXPLODE_FRAMES = [_fly_cell(3, c, FLY_DISPLAY_SIZE) for c in (3, 4)]     # burst
TINY_FRAMES = [_fly_cell(4, c, 24) for c in range(2)]                    # newborn speck

# Frog: first non-empty row of the sheet is the single animation we loop.
frog_sheet = load_image("frog_GameBoy_Green_spritesheet.png")
frog_frames = []
for _row in range(frog_sheet.get_height() // 32):
    frog_frames = sheet_frames(frog_sheet, 32, [_row], range(frog_sheet.get_width() // 32), 56)
    if frog_frames:
        break

# Spider: the lower half of the sheet duplicates the upper half.
spider_sheet = load_image("Spider Sprite Sheet.png")
spider_walk_frames = sheet_frames(spider_sheet, 32, (0, 1, 4, 5, 8, 9, 12, 13),
                                  range(spider_sheet.get_width() // 32), 52)
spider_attack_frames = []
for _row, _cols in ((2, (3, 4, 5)), (10, (3, 4, 5)), (3, (0,)), (11, (0,))):
    spider_attack_frames += sheet_frames(spider_sheet, 32, [_row], _cols, 52)
spider_web_frames = sheet_frames(spider_sheet, 32, (7, 15), range(6), 44)

world_bg = pygame.transform.scale(load_image("basement_7_aseprite_file-frame0.png"), (WORLD_W, H))
darken_overlay = pygame.Surface((WORLD_W, H))
darken_overlay.set_alpha(90)
darken_overlay.fill((10, 10, 15))

# ----------------------------------------------------------------------------
# Brain panel geometry: two half-height slots, one per fly
# ----------------------------------------------------------------------------
ann_full = pd.read_csv("data/flywire_annotations.tsv", sep="\t",
                       usecols=["root_id", "soma_x", "soma_y", "soma_z",
                               "cell_type", "cell_sub_class", "cell_class", "super_class", "side"],
                       low_memory=False)
# A neuron missing soma_x/y can't be placed as a dot, but it's still simulated and can
# still be the "top active neuron" logged in the clinical event log or hand-picked as a
# STIMULI id — so metadata (cell_type etc.) is looked up against the FULL table
# (meta_by_id, below), while `ann` (soma-filtered) is only for what can be drawn/laid out.
ann = ann_full.dropna(subset=["soma_x", "soma_y"])
x_min, x_max = ann["soma_x"].min(), ann["soma_x"].max()
y_min, y_max = ann["soma_y"].min(), ann["soma_y"].max()
z_min, z_max = ann["soma_z"].min(), ann["soma_z"].max()
x_range, y_range = x_max - x_min, y_max - y_min
z_range = max(1.0, z_max - z_min)

# Two brains side by side (not stacked) — each gets the FULL window height
# instead of half of it. That was the actual bottleneck on legibility: with
# neurons scattered across real soma_x/soma_y, the vertical extent was the
# binding constraint on scale, and stacking halved it for no reason.
PAD = 20
SLOT_W = PANEL_W // 2   # one fly's sub-panel width
SLOT_H = H               # one fly's sub-panel height (full window height)
TEXT_BLOCK_H = 410  # vertical space reserved above each brain for the (now single-column) text
BRAIN_AREA_H = SLOT_H - TEXT_BLOCK_H
BRAIN_SCALE = min((SLOT_W - 2 * PAD) / x_range, (BRAIN_AREA_H - 2 * PAD) / y_range)
BRAIN_OFFSET_X = PAD + ((SLOT_W - 2 * PAD) - x_range * BRAIN_SCALE) / 2
BRAIN_OFFSET_Y = TEXT_BLOCK_H + PAD + ((BRAIN_AREA_H - 2 * PAD) - y_range * BRAIN_SCALE) / 2


def slot_coords(x, y):
    """Soma coordinates (arrays) -> pixel coordinates local to one panel slot."""
    sx = BRAIN_OFFSET_X + (np.asarray(x) - x_min) * BRAIN_SCALE
    sy = BRAIN_OFFSET_Y + (np.asarray(y) - y_min) * BRAIN_SCALE
    return sx.astype(np.int32), sy.astype(np.int32)


# Faint dot per soma as the brain silhouette.
brain_bg_half = pygame.Surface((SLOT_W, SLOT_H))
brain_bg_half.fill((0, 0, 0))
_sx, _sy = slot_coords(ann["soma_x"].values, ann["soma_y"].values)
_in_bounds = (_sx >= 0) & (_sx < SLOT_W) & (_sy >= 0) & (_sy < SLOT_H)
_pix = pygame.surfarray.pixels3d(brain_bg_half)
_pix[_sx[_in_bounds], _sy[_in_bounds]] = (50, 50, 85)
del _pix

soma_by_id = {int(k): (x, y, z) for k, x, y, z in
             zip(ann["root_id"].values, ann["soma_x"].values, ann["soma_y"].values, ann["soma_z"].values)}
_layout_cache = {}


def brain_layout(ids):
    """Per-neuron pixel position in a panel slot, a mask of neurons that have one, and a
    depth factor (0=far, 1=near) from the real soma_z (the axis the flat x/y projection
    otherwise throws away). Every LIFBrain has the same neurons, so this is computed once."""
    key = (len(ids), int(ids[0]), int(ids[-1]))
    if key not in _layout_cache:
        coords = np.array([soma_by_id.get(int(nid), (np.nan, np.nan, np.nan)) for nid in ids], dtype=float)
        valid = ~np.isnan(coords[:, 0])
        xy = np.zeros((len(ids), 2), dtype=np.int32)
        xy[valid, 0], xy[valid, 1] = slot_coords(coords[valid, 0], coords[valid, 1])
        depth = np.full(len(ids), 0.5, dtype=np.float32)
        depth[valid] = 1.0 - ((coords[valid, 2] - z_min) / z_range)
        _layout_cache[key] = (xy, valid, depth)
    return _layout_cache[key]


def _s(v):
    return str(v) if pd.notna(v) else ""


meta_by_id = {
    int(k): (_s(ct), _s(sc), _s(cc), _s(sup))
    for k, ct, sc, cc, sup in zip(ann_full["root_id"].values, ann_full["cell_type"].values,
                                  ann_full["cell_sub_class"].values, ann_full["cell_class"].values,
                                  ann_full["super_class"].values)
}

# Body side ("left"/"right"/"center"/"na") for every neuron — used to split the real
# descending-neuron population by side for steering decode (see descending_lr below).
side_by_id = {int(k): _s(v) for k, v in zip(ann_full["root_id"].values, ann_full["side"].values)}


def neuron_meta(root_id):
    """(cell_type, cell_sub_class, super_class) for the hover tooltip. Real FlyWire
    annotation fields, not anything derived — '?'/'' when the table has no value."""
    ct, sc, cc, sup = meta_by_id.get(int(root_id), ("", "", "", ""))
    return (ct or "?"), sc, sup


def neuron_display_type(root_id):
    """A label for the event log that's never a bare '?'. Most FlyWire neurons DO
    have a fine cell_type (e.g. 'PAM02', 'LC4'), but many — especially in less-studied
    regions — only have the coarser cell_class or super_class filled in. Falling back
    through cell_type -> cell_class -> super_class means the log always shows the most
    specific real label the table actually has, instead of a dead-end '?'."""
    ct, sc, cc, sup = meta_by_id.get(int(root_id), ("", "", "", ""))
    if ct:
        return ct, False
    if cc:
        return cc, True
    if sup:
        return sup, True
    return "unlabeled", True


_trail_cache = {}


def trail_targets(ids, W):
    """For each neuron, its top-K real downstream synaptic targets by |weight| (row indices
    into the same brain). Used to draw a spike trail only along an actual synapse — between
    two neurons that are BOTH currently firing hard — instead of an arbitrary decoration.
    Computed once per subgraph and cached, since every fly shares the same connectome."""
    key = (len(ids), int(ids[0]), int(ids[-1]))
    if key not in _trail_cache:
        coo = W.tocoo()  # rows=post(target), cols=pre(source)
        edges = pd.DataFrame({"pre": coo.col, "post": coo.row, "w": np.abs(coo.data)})
        top = edges.sort_values("w", ascending=False).groupby("pre").head(TRAIL_TARGETS_PER_NEURON)
        N = len(ids)
        targets = np.full((N, TRAIL_TARGETS_PER_NEURON), -1, dtype=np.int32)
        counts = np.zeros(N, dtype=np.int32)
        for pre, post in zip(top["pre"].values, top["post"].values):
            c = counts[pre]
            if c < TRAIL_TARGETS_PER_NEURON:
                targets[pre, c] = post
                counts[pre] = c + 1
        _trail_cache[key] = targets
    return _trail_cache[key]


ALL_STIMULI_IDS = set().union(*(cfg["ids"] for cfg in STIMULI.values()))
_identity_cache = {}


def neuron_identity(ids):
    """Per-neuron (category, is_core) for one brain's neuron list. category
    comes from NEURON_OWNER (falls back to ambient if unmapped); is_core
    marks the literal hand-picked STIMULI neurons, drawn larger than the
    neurons the subgraph recruited around them. Same subgraph for every fly,
    so this is computed once and cached like brain_layout."""
    key = (len(ids), int(ids[0]), int(ids[-1]))
    if key not in _identity_cache:
        category = np.array([NEURON_OWNER.get(int(nid), CAT_AMBIENT) for nid in ids], dtype=np.int8)
        core = np.array([int(nid) in ALL_STIMULI_IDS for nid in ids], dtype=bool)
        _identity_cache[key] = (category, core)
    return _identity_cache[key]


_descending_cache = {}


def descending_lr(ids):
    """Local indices (within this brain's neuron list) of the real descending
    neurons, split left/right by actual anatomy. Descending neurons are the axons
    that carry commands OUT of the central brain toward the (missing, in this sim)
    nerve cord — they're recruited into the subgraph naturally as downstream
    targets of vision/motion/fear/etc, not driven directly, so there's no STIMULI
    entry for them; this just locates whichever ones ended up in a given brain.
    Same subgraph for every fly, so cached like neuron_identity/brain_layout."""
    key = (len(ids), int(ids[0]), int(ids[-1]))
    if key not in _descending_cache:
        left_idx, right_idx = [], []
        for i, nid in enumerate(ids):
            nid = int(nid)
            if meta_by_id.get(nid, ("", "", "", ""))[3] != "descending":
                continue
            side = side_by_id.get(nid, "")
            if side == "left":
                left_idx.append(i)
            elif side == "right":
                right_idx.append(i)
        _descending_cache[key] = (np.array(left_idx, dtype=int), np.array(right_idx, dtype=int))
    return _descending_cache[key]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def random_point(margin=30):
    return np.array([random.uniform(margin, WORLD_W - margin), random.uniform(margin, H - margin)])


def mean_at(values, idx):
    return float(values[idx].mean()) if len(idx) else 0.0


def compute_vision(pos, vel, weighted_objects, sight_radius):
    """Left/right visual drive from weighted objects within sight radius."""
    heading = vel / (np.linalg.norm(vel) + 1e-6)
    left_signal, right_signal = 0.0, 0.0
    for obj_pos, weight in weighted_objects:
        to_obj = obj_pos - pos
        dist = np.linalg.norm(to_obj)
        if dist < 1e-6 or dist > sight_radius:
            continue
        to_obj_norm = to_obj / dist
        cross_z = heading[0] * to_obj_norm[1] - heading[1] * to_obj_norm[0]
        closeness = (1.0 - dist / sight_radius) * weight
        if cross_z > 0:
            right_signal += closeness
        else:
            left_signal += closeness
    return left_signal, right_signal


# ----------------------------------------------------------------------------
# Genetics
# ----------------------------------------------------------------------------
mutation_events = 0  # one event = one offspring genome from mutate_genome()


def random_genome():
    return {"eye_hue": random.uniform(0, 360), "turn_gain": random.uniform(0.10, 0.20),
            "sight_radius": random.uniform(150, 220), "base_speed": random.uniform(1.7, 2.3)}


def mutate_genome(g1, g2):
    global mutation_events
    mutation_events += 1
    child = {}
    for key in GENOME_TRAITS:
        blend = (g1[key] + g2[key]) / 2
        if key == "eye_hue":
            child[key] = (blend + random.uniform(-20, 20)) % 360
        else:
            child[key] = blend * random.uniform(0.9, 1.1)
    child["eye_hue_parents"] = (g1["eye_hue"], g2["eye_hue"])
    return child


# ----------------------------------------------------------------------------
# Fly
# ----------------------------------------------------------------------------
class Fly:
    def __init__(self, pos, genome, fly_id, birth_time_ms=0):
        self.fly_id = fly_id
        self.brain = LIFBrain(dt=1.0)
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.vel /= np.linalg.norm(self.vel)
        self.genome = genome
        self.marked = genome.get("marked", False)  # the one the frog hunts
        self.birth_time_ms = birth_time_ms  # for the necropsy card and lineage record
        self.doomed = False
        self.current_speed = genome["base_speed"]
        self.frames = [
            pygame.transform.scale(recolor_eyes(f, genome["eye_hue"]), (FLY_DISPLAY_SIZE, FLY_DISPLAY_SIZE))
            for f in fly_frames_raw
        ]
        self.anim_frame, self.anim_timer = 0, 0
        self.webbed_ms = 0
        self.mating_cooldown = 300

        parents = genome.get("eye_hue_parents")
        if parents is not None:
            print(f"[mutation] FLY {fly_id:02d} eye hue: parents {parents[0]:.1f}° & {parents[1]:.1f}° "
                  f"-> {genome['eye_hue']:.1f}°")
        else:
            print(f"[mutation] FLY {fly_id:02d} eye hue: {genome['eye_hue']:.1f}° (original stock)")

        # --- neuron populations (ids for driving the brain, indices for reading it) ---
        self.sugar_ids, sugar_idx = self._population("sugar")
        self.water_ids, water_idx = self._population("water")
        self.salt_ids, salt_idx = self._population("salt")
        self.fear_ids, self.fear_idx = self._population("fear")
        self.motion_ids, motion_idx = self._population("motion")
        self.vision_left_ids, self.vision_left_idx = self._population("vision_left")
        self.vision_right_ids, self.vision_right_idx = self._population("vision_right")
        self.mating_ids, self.mating_idx = self._population("mating")
        self.reward_ids, self.reward_idx = self._population("dopamine_reward")
        self.punish_ids, self.punish_idx = self._population("dopamine_punish")
        self.olfactory_left_ids, self.olfactory_left_idx = self._population("olfactory_left")
        self.olfactory_right_ids, self.olfactory_right_idx = self._population("olfactory_right")
        # bitter/aggression aren't driven by anything in this sim yet and aren't read
        # individually — their neurons still get identified and colored via
        # NEURON_OWNER/neuron_identity() below, from the subgraph build, not from here.

        # Descending neurons aren't driven directly (no STIMULI entry) — they're the
        # real output population, recruited into the subgraph as downstream targets
        # of everything else. Steering is decoded from THEIR firing, not the eyes'.
        self.desc_left_idx, self.desc_right_idx = descending_lr(self.brain.ids)
        self.recent_desc_left = 0.0
        self.recent_desc_right = 0.0
        self.display_desc_left = 0.0
        self.display_desc_right = 0.0
        self.display_smell_left = 0.0
        self.display_smell_right = 0.0
        self.total_activity = 0.0  # smoothed mean firing rate across the whole brain, for the drone

        self.feed_ids = {"sugar": self.sugar_ids, "water": self.water_ids, "salt": self.salt_ids}
        self.feed_event_indices = {"sugar": sugar_idx, "water": water_idx, "salt": salt_idx}

        # Every neuron's identity color: which behavior structurally recruits
        # it the hardest (see NEURON_OWNER / build_subgraph.py), not just the
        # handful of neurons we directly drive. Same subgraph for every fly,
        # so this (and whether a neuron is one of the literal hand-picked
        # STIMULI ids, drawn slightly larger) is computed once and cached.
        self.neuron_category, self.core_mask = neuron_identity(self.brain.ids)

        # --- brain-panel state ---
        self.slot_xy, self.slot_valid, self.slot_depth = brain_layout(self.brain.ids)
        self.trail_targets = trail_targets(self.brain.ids, self.brain.W)
        self.category_glow = np.zeros(self.brain.N, dtype=np.float32)  # decaying spike trace, drives brightness
        self.recent_left = 0.0    # decaying vision spike trace, drives steering
        self.recent_right = 0.0
        self.display_visual_left = 0.0
        self.display_visual_right = 0.0
        self.mating_activity = 0.0
        self.fear_activity = 0.0
        self.reward_activity = 0.0
        self.punish_activity = 0.0
        self.active_categories = np.zeros(len(CATEGORY_COLORS), dtype=bool)
        self.active_categories[CAT_AMBIENT] = True

        # --- feeding event ---
        self.feed_event = None
        self.feed_event_remaining_ms = 0.0
        self.feed_event_spikes = 0
        self.feed_event_neurons = 0
        self.feed_event_announce = False

    def _population(self, name):
        ids = [nid for nid in STIMULI[name]["ids"] if nid in self.brain.id_to_idx]
        idx = np.array([self.brain.id_to_idx[nid] for nid in ids], dtype=int)
        return ids, idx

    # --- feeding -------------------------------------------------------------
    def feed_event_active(self):
        return self.feed_event is not None and self.feed_event_remaining_ms > 0

    def start_feed_event(self, kind):
        self.feed_event = kind
        self.feed_event_remaining_ms = float(FEED_EVENT_DURATION_MS)
        self.feed_event_spikes = 0
        self.feed_event_neurons = len(self.feed_event_indices[kind])
        self.feed_event_announce = True

    def _update_feeding(self, fruits, waters, salts, dt_ms):
        # A feeding event is discrete: it is not restarted while active.
        if not self.feed_event_active():
            for kind, items in (("sugar", fruits), ("water", waters), ("salt", salts)):
                item = next((i for i in items if np.linalg.norm(self.pos - i["pos"]) < FEED_PICKUP_DIST), None)
                if item is not None:
                    self.start_feed_event(kind)
                    item["pos"] = random_point()
                    if kind == "sugar":
                        item["sprite"] = random.choice(fruit_sprites)
                    break

        if self.feed_event is not None:
            self.feed_event_remaining_ms = max(0.0, self.feed_event_remaining_ms - dt_ms)
            if self.feed_event_remaining_ms <= 0:
                self.feed_event = None
                self.feed_event_spikes = 0
                self.feed_event_neurons = 0

    # --- per-frame update ----------------------------------------------------
    def update(self, fruits, frog, waters, salts, spiders, other_flies, mating_season, dance_angle, dt_ms):
        protected = mating_season and not self.marked
        mating_target = None
        if mating_season:
            mating_target = min((f for f in other_flies if f is not self),
                                key=lambda f: np.linalg.norm(f.pos - self.pos), default=None)

        self._move(dt_ms)
        self._update_feeding(fruits, waters, salts, dt_ms)
        fear_hz = self._fear_and_escape(frog, protected)

        visible = [(f["pos"], 1.0) for f in fruits]
        visible += [(w["pos"], 0.8) for w in waters]
        visible += [(s["pos"], 0.8) for s in salts]
        if not protected:
            visible.append((frog.pos, -1.5))
            visible += [(sp.pos, -1.8) for sp in spiders]
        vl, vr = compute_vision(self.pos, self.vel, visible, self.genome["sight_radius"])

        # Smell: the same food sources vision picks up, but world-spanning range
        # instead of the genome's (much shorter) sight_radius — food behind the fly
        # or across the room still registers, weakly, the way real olfaction isn't
        # limited to a forward cone of view.
        food_odor = [(f["pos"], 1.0) for f in fruits] + [(w["pos"], 0.8) for w in waters] + \
                    [(s["pos"], 0.8) for s in salts]
        sl, sr = compute_vision(self.pos, self.vel, food_odor, SMELL_RADIUS)

        # Courtship drive grows as the mating target gets closer.
        mating_hz = 0.0
        if mating_target is not None:
            dist = np.linalg.norm(mating_target.pos - self.pos)
            if dist < MATING_COURTING_RADIUS:
                mating_hz = STIMULI["mating"]["hz"] * (1.0 - dist / MATING_COURTING_RADIUS)

        # PAM (reward) fires on the reward itself — landing a feed event — not on
        # searching for food. PPL1 (punish) fires on the aversive event actually
        # landing: webbed, or a frog close enough to be a real threat (fear_hz>0),
        # not just ambient wariness.
        reward_hz = STIMULI["dopamine_reward"]["hz"] if self.feed_event_active() else 0.0
        punish_hz = STIMULI["dopamine_punish"]["hz"] if (self.webbed_ms > 0 or fear_hz > 0) else 0.0

        frame_spikes = self._run_brain(vl, vr, fear_hz, mating_hz, reward_hz, punish_hz, sl, sr)
        self._update_readouts(frame_spikes)
        self._steer(mating_target, dance_angle)

    def _move(self, dt_ms):
        if self.webbed_ms > 0:
            self.webbed_ms = max(0, self.webbed_ms - dt_ms)
            self.current_speed = 0.0
        else:
            self.pos += self.vel * self.current_speed
            self.current_speed = max(self.genome["base_speed"], self.current_speed * 0.95)
        if self.pos[0] < 0 or self.pos[0] > WORLD_W:
            self.vel[0] *= -1
        if self.pos[1] < 0 or self.pos[1] > H:
            self.vel[1] *= -1
        self.pos = np.clip(self.pos, 0, [WORLD_W, H])

        self.anim_timer += 1
        if self.anim_timer >= 6:
            self.anim_frame = (self.anim_frame + 1) % len(self.frames)
            self.anim_timer = 0
        if self.mating_cooldown > 0:
            self.mating_cooldown -= 1

    def _fear_and_escape(self, frog, protected):
        """Fear drive (Hz) for the LC4/LPLC2 population; also steers away from the frog.
        Spiders only cause fear once their web actually hits (webbed = max fear)."""
        if self.webbed_ms > 0:
            return FEAR_HZ
        frog_dist = np.linalg.norm(self.pos - frog.pos)
        if protected or frog_dist >= FROG_FEAR_RADIUS:
            return 0.0
        intensity = 1.0 - frog_dist / FROG_FEAR_RADIUS
        away = self.pos - frog.pos
        d = np.linalg.norm(away)
        if d > 1e-6:
            strength = 0.08 + 0.20 * intensity
            self.vel = self.vel * (1.0 - strength) + away / d * strength
            self.current_speed = max(self.current_speed, self.genome["base_speed"] * (1.0 + 1.8 * intensity))
        return FEAR_HZ * intensity

    def _run_brain(self, vl, vr, fear_hz, mating_hz, reward_hz=0.0, punish_hz=0.0, sl=0.0, sr=0.0):
        """Run the substeps for this frame and update the panel glow. Returns spike counts per neuron."""
        drive = {}
        if self.feed_event_active():
            drive.update(dict.fromkeys(self.feed_ids[self.feed_event], STIMULI[self.feed_event]["hz"]))
        if fear_hz > 0:
            drive.update(dict.fromkeys(self.fear_ids, fear_hz))
        motion_hz = STIMULI["motion"]["hz"] * (self.current_speed / self.genome["base_speed"])
        drive.update(dict.fromkeys(self.motion_ids, motion_hz))
        drive.update(dict.fromkeys(self.vision_left_ids, STIMULI["vision_left"]["hz"] * vl))
        drive.update(dict.fromkeys(self.vision_right_ids, STIMULI["vision_right"]["hz"] * vr))
        drive.update(dict.fromkeys(self.olfactory_left_ids, STIMULI["olfactory_left"]["hz"] * sl))
        drive.update(dict.fromkeys(self.olfactory_right_ids, STIMULI["olfactory_right"]["hz"] * sr))
        if mating_hz > 0:
            drive.update(dict.fromkeys(self.mating_ids, mating_hz))
        if reward_hz > 0:
            drive.update(dict.fromkeys(self.reward_ids, reward_hz))
        if punish_hz > 0:
            drive.update(dict.fromkeys(self.punish_ids, punish_hz))

        frame_spikes = np.zeros(self.brain.N, dtype=np.float32)
        for _ in range(STEPS_PER_FRAME):
            spiked = np.asarray(self.brain.step(drive), dtype=np.float32)
            self.recent_left = self.recent_left * 0.95 + mean_at(spiked, self.vision_left_idx)
            self.recent_right = self.recent_right * 0.95 + mean_at(spiked, self.vision_right_idx)
            # This is the real steering signal — the actual descending-neuron output,
            # not the eyes. It naturally blends vision + smell + everything else,
            # because that's what the real synaptic network upstream of these neurons
            # already does; nothing here hand-picks which sense "counts".
            self.recent_desc_left = self.recent_desc_left * 0.95 + mean_at(spiked, self.desc_left_idx)
            self.recent_desc_right = self.recent_desc_right * 0.95 + mean_at(spiked, self.desc_right_idx)
            self.category_glow *= GLOW_DECAY
            self.category_glow += spiked
            frame_spikes += spiked

        return frame_spikes

    def _update_readouts(self, frame_spikes):
        rate = frame_spikes / STEPS_PER_FRAME
        def smooth(old, new):
            return old * DISPLAY_DECAY + new * (1 - DISPLAY_DECAY)
        self.display_visual_left = smooth(self.display_visual_left, mean_at(rate, self.vision_left_idx))
        self.display_visual_right = smooth(self.display_visual_right, mean_at(rate, self.vision_right_idx))
        self.display_smell_left = smooth(self.display_smell_left, mean_at(rate, self.olfactory_left_idx))
        self.display_smell_right = smooth(self.display_smell_right, mean_at(rate, self.olfactory_right_idx))
        self.display_desc_left = smooth(self.display_desc_left, mean_at(rate, self.desc_left_idx))
        self.display_desc_right = smooth(self.display_desc_right, mean_at(rate, self.desc_right_idx))
        self.total_activity = smooth(self.total_activity, float(rate.mean()))
        self.mating_activity = smooth(self.mating_activity, mean_at(rate, self.mating_idx))
        self.fear_activity = smooth(self.fear_activity, mean_at(rate, self.fear_idx))
        self.reward_activity = smooth(self.reward_activity, mean_at(rate, self.reward_idx))
        self.punish_activity = smooth(self.punish_activity, mean_at(rate, self.punish_idx))

        # Which behaviors are actually live right now — used to decide whether a
        # *recruited* neuron's firing gets shown in its owner's color (draw_brain_dots)
        # or folded into ambient, so the map never implies e.g. "fear" when FEAR reads 0.
        self.active_categories[CAT_FEAR] = self.fear_activity > CATEGORY_ACTIVE_THRESHOLD
        self.active_categories[CAT_MATING] = self.mating_activity > CATEGORY_ACTIVE_THRESHOLD
        self.active_categories[CAT_REWARD] = self.reward_activity > CATEGORY_ACTIVE_THRESHOLD
        self.active_categories[CAT_PUNISH] = self.punish_activity > CATEGORY_ACTIVE_THRESHOLD
        for kind, cat in FEED_CATEGORY.items():
            self.active_categories[cat] = self.feed_event == kind and self.feed_event_active()

        if self.feed_event_active():
            self.feed_event_spikes = int(frame_spikes[self.feed_event_indices[self.feed_event]].sum())
            if self.feed_event_announce:
                print(f"FLY {self.fly_id:02d} {FEED_EVENT_LABELS[self.feed_event]} | "
                      f"neurons={self.feed_event_neurons} | spikes={self.feed_event_spikes}")
                self.feed_event_announce = False

    def _steer(self, mating_target, dance_angle):
        # Descending-neuron steering: turn is decoded from the real output
        # population that carries commands out of the central brain (recent_left/
        # right, the eyes' own activity, are kept only for the VISUAL LEFT/RIGHT
        # panel readout below — they no longer drive movement directly).
        turn = (self.recent_desc_right - self.recent_desc_left) * self.genome["turn_gain"]
        c, s = np.cos(turn), np.sin(turn)
        self.vel = np.array([self.vel[0] * c - self.vel[1] * s, self.vel[0] * s + self.vel[1] * c])

        # Mating season: drift toward the nearest fly, then circle it once close.
        if mating_target is not None:
            to_mate = mating_target.pos - self.pos
            mate_dist = np.linalg.norm(to_mate)
            if mate_dist > 1e-6:
                if mate_dist > ORBIT_ENTER_DIST:
                    attraction = min(MATING_ATTRACTION_MAX,
                                     MATING_ATTRACTION_STRENGTH * (1.0 + 80.0 / max(mate_dist, 80.0)))
                    self.vel = self.vel * (1.0 - attraction) + to_mate / mate_dist * attraction
                else:
                    midpoint = (self.pos + mating_target.pos) / 2.0
                    phase = 0.0 if self.fly_id < mating_target.fly_id else np.pi
                    target_point = midpoint + ORBIT_RADIUS * np.array([np.cos(dance_angle + phase),
                                                                       np.sin(dance_angle + phase)])
                    to_target = target_point - self.pos
                    td = np.linalg.norm(to_target)
                    if td > 1e-6:
                        self.vel = self.vel * (1.0 - ORBIT_STEER_STRENGTH) + to_target / td * ORBIT_STEER_STRENGTH

        self.vel /= np.linalg.norm(self.vel)

    def top_active_neuron(self):
        """The hand-picked "core" neuron currently glowing hardest, for the clinical
        event log — real root_id, real cell_type, sampled off actual spike activity,
        not synthesized. Ambient (motion/vision) neurons are excluded on purpose: they
        fire almost continuously just from walking around, and would flood the log
        with 'AMBIENT' rows that drown out the behaviorally meaningful ones (feed,
        fear, mating, reward, punish). None while nothing behavioral is firing.

        Also gated on active_categories — the same live-check draw_brain_dots and the
        panel badges use — not just raw per-neuron glow. dopamine_punish is only 16
        neurons; without this gate, one of them sitting with residual glow just above
        GLOW_THRESHOLD could win the argmax and flood the log with PUNISH rows even
        while the smoothed PUNISH (PPL1) readout stays under the 'actually active'
        threshold and the panel shows no purple at all — a real neuron firing a
        little, misreported as the circuit being live."""
        live_mask = self.active_categories[self.neuron_category]
        eligible = (self.core_mask & (self.neuron_category != CAT_AMBIENT) & live_mask &
                    (self.category_glow > GLOW_THRESHOLD))
        idx = np.flatnonzero(eligible)
        if not idx.size:
            return None
        best = int(idx[np.argmax(self.category_glow[idx])])
        return (int(self.brain.ids[best]), int(self.neuron_category[best]),
               float(self.category_glow[best]))

    def panel_state(self):
        """Everything the brain panel needs. Also used as the frozen 'mating snapshot'."""
        return {
            "fly_id": self.fly_id, "snapshot": False,
            "glow": self.category_glow, "categories": self.neuron_category, "core": self.core_mask,
            "active": self.active_categories,
            "slot_xy": self.slot_xy, "slot_valid": self.slot_valid, "slot_depth": self.slot_depth,
            "ids": self.brain.ids, "trail_targets": self.trail_targets,
            "left": self.display_visual_left, "right": self.display_visual_right,
            "smell_left": self.display_smell_left, "smell_right": self.display_smell_right,
            "desc_left": self.display_desc_left, "desc_right": self.display_desc_right,
            "fear": self.fear_activity, "mating": self.mating_activity,
            "reward": self.reward_activity, "punish": self.punish_activity,
            "feed_event": self.feed_event, "feed_remaining_ms": self.feed_event_remaining_ms,
            "feed_spikes": self.feed_event_spikes, "feed_neurons": self.feed_event_neurons,
            "feed_indices": self.feed_event_indices,
        }

    def draw(self, surf):
        img = self.frames[self.anim_frame]
        if self.vel[0] < 0:
            img = pygame.transform.flip(img, True, False)
        surf.blit(img, img.get_rect(center=self.pos.astype(int)))
        if self.webbed_ms > 0 and spider_web_frames:
            web_img = spider_web_frames[-1]
            surf.blit(web_img, web_img.get_rect(center=self.pos.astype(int)))


# ----------------------------------------------------------------------------
# Life-cycle objects
# ----------------------------------------------------------------------------
class Metamorphosis:
    """Grow-then-explode (mating) or grow-only (hatching) animation."""
    FRAME_HOLD = 8

    def __init__(self, pos, genome, mode):
        self.pos = np.array(pos, dtype=float)
        self.genome = genome
        self.mode = mode  # "merge_explode" or "hatch"
        self.frame_timer = 0
        self.grow_idx = 0
        self.explode_idx = 0
        self.stage = "grow"

    def update(self):
        self.frame_timer += 1
        if self.frame_timer < self.FRAME_HOLD:
            return
        self.frame_timer = 0
        if self.stage == "grow":
            self.grow_idx += 1
            if self.grow_idx >= len(GROW_FRAMES):
                self.stage = "explode" if self.mode == "merge_explode" else "done"
        elif self.stage == "explode":
            self.explode_idx += 1
            if self.explode_idx >= len(EXPLODE_FRAMES):
                self.stage = "done"

    @property
    def finished(self):
        return self.stage == "done"

    def draw(self, surf):
        if self.stage == "grow":
            img = GROW_FRAMES[min(self.grow_idx, len(GROW_FRAMES) - 1)]
        elif self.stage == "explode":
            img = EXPLODE_FRAMES[min(self.explode_idx, len(EXPLODE_FRAMES) - 1)]
        else:
            return
        surf.blit(img, img.get_rect(center=self.pos.astype(int)))


class Maggot:
    def __init__(self, pos, genome, duration=600):
        self.pos = np.array(pos, dtype=float)
        self.genome = genome
        self.timer = duration
        self.anim_timer, self.anim_frame = 0, 0

    def update(self):
        self.timer -= 1
        self.anim_timer += 1
        if self.anim_timer >= 15:
            self.anim_timer = 0
            self.anim_frame = (self.anim_frame + 1) % len(TINY_FRAMES)

    def draw(self, surf):
        img = TINY_FRAMES[self.anim_frame]
        surf.blit(img, img.get_rect(center=self.pos.astype(int)))


class Heart:
    """Animated heart that pops out of a courting pair and drifts upward.
    Starts as the full heart gif; switches to the broken one when the flies mate."""
    def __init__(self, origin):
        self.pos = np.array(origin, dtype=float)
        angle = random.uniform(0, np.pi * 2)
        speed = random.uniform(0.9, 2.1)
        self.vel = np.array([np.cos(angle) * speed, np.sin(angle) * speed * 0.7 - 0.5])
        self.life = random.randint(150, 260)
        self.phase = random.uniform(0, np.pi * 2)
        self.scale = random.uniform(0.6, 1.0)
        self.damaged = False
        self.frame_timer = random.uniform(0, HEART_FRAME_MS)
        self.frame_idx = random.randint(0, 4)
        self._scaled = {}

    def _frames(self):
        return heart_damaged_frames if self.damaged else heart_full_frames

    def update(self, dt_ms=16):
        self.pos += self.vel
        self.vel[0] += np.sin(self.phase + self.life * 0.035) * 0.002
        self.life -= 1
        # Ease out of the initial pop into a gentle upward drift.
        self.vel[0] *= 0.985
        self.vel[1] = self.vel[1] * 0.985 - 0.004
        frames = self._frames()
        if frames:
            self.frame_timer += dt_ms
            if self.frame_timer >= HEART_FRAME_MS:
                self.frame_timer = 0
                self.frame_idx = (self.frame_idx + 1) % len(frames)

    @property
    def finished(self):
        return self.life <= 0 or self.pos[1] < -30

    def draw(self, surf):
        frames = self._frames()
        if not frames:
            return
        base = frames[self.frame_idx % len(frames)]
        img = self._scaled.get(id(base))
        if img is None:
            size = (max(8, int(base.get_width() * self.scale)), max(8, int(base.get_height() * self.scale)))
            img = self._scaled[id(base)] = pygame.transform.smoothscale(base, size)
        img.set_alpha(int(255 * min(1.0, self.life / 35.0)))
        surf.blit(img, img.get_rect(center=self.pos.astype(int)))


# ----------------------------------------------------------------------------
# Predators
# ----------------------------------------------------------------------------
predator_stats = {"web_shots": 0, "web_hits": 0}


class WebProjectile:
    def __init__(self, pos, target):
        self.pos = np.array(pos, dtype=float)
        direction = target.pos - self.pos
        self.vel = direction / max(np.linalg.norm(direction), 1e-6) * WEB_SPEED
        self.target = target
        self.life_ms = 1800
        self.frame_idx = 0
        self.frame_timer = 0
        self.hit = False

    def update(self, dt_ms):
        self.life_ms -= dt_ms
        if self.life_ms <= 0 or self.hit:
            return
        self.pos += self.vel
        self.frame_timer += dt_ms
        if self.frame_timer >= 80:
            self.frame_timer = 0
            self.frame_idx = (self.frame_idx + 1) % max(1, len(spider_web_frames))

        if self.target.webbed_ms <= 0 and np.linalg.norm(self.pos - self.target.pos) < 22:
            self.target.webbed_ms = WEB_STICK_MS
            if self.target.marked:
                self.target.doomed = True   # wrapped and taken once the web expires
            self.hit = True
            predator_stats["web_hits"] += 1

    @property
    def finished(self):
        return self.life_ms <= 0 or self.hit

    def draw(self, surf):
        if spider_web_frames:
            img = spider_web_frames[self.frame_idx % len(spider_web_frames)]
            surf.blit(img, img.get_rect(center=self.pos.astype(int)))


class Spider:
    """Roaming ambush predator. It never chases; it only shoots webs at flies in range."""
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.vel /= np.linalg.norm(self.vel) + 1e-6
        self.anim_frame = 0
        self.anim_timer = 0
        self.attack_timer_ms = random.randint(700, SPIDER_SHOT_INTERVAL_MS)
        self.attack_flash_ms = 0
        self.turn_timer_ms = random.randint(300, SPIDER_TURN_INTERVAL_MS)

    def update(self, flies, webs, dt_ms, mating_season):
        self.attack_timer_ms = max(0, self.attack_timer_ms - dt_ms)
        self.attack_flash_ms = max(0, self.attack_flash_ms - dt_ms)
        self.turn_timer_ms = max(0, self.turn_timer_ms - dt_ms)

        if self.turn_timer_ms <= 0:
            angle = random.uniform(-0.9, 0.9)
            c, s = np.cos(angle), np.sin(angle)
            self.vel = np.array([self.vel[0] * c - self.vel[1] * s, self.vel[0] * s + self.vel[1] * c])
            self.vel /= np.linalg.norm(self.vel) + 1e-6
            self.turn_timer_ms = random.randint(700, SPIDER_TURN_INTERVAL_MS)

        living = [f for f in flies if f.webbed_ms <= 0 and not (mating_season and not f.marked)]
        target = min(living, key=lambda f: np.linalg.norm(f.pos - self.pos), default=None)
        if target is not None and self.attack_timer_ms <= 0:
            if np.linalg.norm(target.pos - self.pos) <= SPIDER_WEB_RANGE:
                webs.append(WebProjectile(self.pos, target))
                self.attack_timer_ms = SPIDER_SHOT_INTERVAL_MS
                self.attack_flash_ms = SPIDER_ATTACK_FLASH_MS
                predator_stats["web_shots"] += 1

        self.pos += self.vel * SPIDER_SPEED
        if self.pos[0] < 25 or self.pos[0] > WORLD_W - 25:
            self.vel[0] *= -1
        if self.pos[1] < 25 or self.pos[1] > H - 25:
            self.vel[1] *= -1
        self.pos = np.clip(self.pos, 20, [WORLD_W - 20, H - 20])

        self.anim_timer += 1
        if self.anim_timer >= 7:
            self.anim_timer = 0
            self.anim_frame = (self.anim_frame + 1) % max(1, len(spider_walk_frames))

    def draw(self, surf):
        if self.attack_flash_ms > 0 and spider_attack_frames:
            idx = min(len(spider_attack_frames) - 1, int((SPIDER_ATTACK_FLASH_MS - self.attack_flash_ms) / 75))
            img = spider_attack_frames[idx]
        else:
            img = spider_walk_frames[self.anim_frame % len(spider_walk_frames)]
        surf.blit(img, img.get_rect(center=self.pos.astype(int)))


class Frog:
    """Patrolling frog: hops between spots and chases a nearby fly for at most 5 seconds.
    The marked fly is hunted from anywhere, faster than it can run, and eaten on contact."""
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.target = self.pos.copy()
        self.chase_target = None
        self.chase_timer_ms = 0
        self.chase_cooldown_ms = 0
        self.ate = None
        self.hop_timer_ms = 0
        self.anim_frame = 0
        self.anim_timer = 0
        self.choose_hop_target()

    def choose_hop_target(self):
        angle = random.uniform(0, np.pi * 2)
        distance = random.uniform(FROG_JUMP_DISTANCE_MIN, FROG_JUMP_DISTANCE_MAX)
        self.target = np.clip(self.pos + np.array([np.cos(angle), np.sin(angle)]) * distance,
                              [35, 35], [WORLD_W - 35, H - 35])
        self.hop_timer_ms = FROG_JUMP_PAUSE_MS

    def update(self, flies, dt_ms, mating_season):
        self.chase_cooldown_ms = max(0, self.chase_cooldown_ms - dt_ms)

        # Keep chasing only while the fly is in range and the timer lasts (marked flies never escape).
        if self.chase_target is not None:
            if self.chase_target not in flies:
                self.chase_target = None
                self.chase_timer_ms = 0
            elif not self.chase_target.marked:
                self.chase_timer_ms -= dt_ms
                if (np.linalg.norm(self.chase_target.pos - self.pos) > FROG_DETECTION_RADIUS
                        or self.chase_timer_ms <= 0):
                    self.chase_target = None
                    self.chase_timer_ms = 0
                    self.chase_cooldown_ms = FROG_CHASE_COOLDOWN_MS

        if self.chase_target is None and self.chase_cooldown_ms <= 0:
            marked = [f for f in flies if f.marked]
            if marked:
                self.chase_target = min(marked, key=lambda f: np.linalg.norm(f.pos - self.pos))
                self.chase_timer_ms = FROG_CHASE_DURATION_MS
            else:
                candidates = [f for f in flies if not (mating_season and not f.marked)]
                nearest = min(candidates, key=lambda f: np.linalg.norm(f.pos - self.pos), default=None)
                if nearest is not None and np.linalg.norm(nearest.pos - self.pos) < FROG_DETECTION_RADIUS:
                    self.chase_target = nearest
                    self.chase_timer_ms = FROG_CHASE_DURATION_MS

        if self.chase_target is not None:
            speed = FROG_CHASE_SPEED * 2.2 if self.chase_target.marked else FROG_CHASE_SPEED
            direction = self.chase_target.pos - self.pos
            d = np.linalg.norm(direction)
            if d > 1e-6:
                self.pos += direction / d * speed
        else:
            self.hop_timer_ms = max(0, self.hop_timer_ms - dt_ms)
            direction = self.target - self.pos
            d = np.linalg.norm(direction)
            if d < 10:
                self.choose_hop_target()
            elif self.hop_timer_ms <= 0:
                self.pos += direction / d * FROG_PATROL_SPEED

        self.ate = None
        if (self.chase_target is not None and self.chase_target.marked
                and np.linalg.norm(self.chase_target.pos - self.pos) < 26):
            self.ate = self.chase_target
            self.chase_target = None
            self.chase_timer_ms = 0

        self.anim_timer += 1
        if self.anim_timer >= 6 and frog_frames:
            self.anim_timer = 0
            self.anim_frame = (self.anim_frame + 1) % len(frog_frames)

    def draw(self, surf):
        if not frog_frames:
            return
        img = frog_frames[self.anim_frame % len(frog_frames)]
        if self.chase_target is not None and self.chase_target.pos[0] < self.pos[0]:
            img = pygame.transform.flip(img, True, False)
        surf.blit(img, img.get_rect(center=self.pos.astype(int)))


# ----------------------------------------------------------------------------
# World state
# ----------------------------------------------------------------------------
def random_fruit():
    return {"pos": random_point(), "sprite": random.choice(fruit_sprites)}


def random_water():
    return {"pos": random_point()}


def random_salt():
    return {"pos": random_point(), "sprite": random.choice(salt_sprites)}


fruits = [random_fruit() for _ in range(8)]
waters = [random_water() for _ in range(3)]
salts = [random_salt() for _ in range(3)]
frog = Frog(random_point())
spiders = [Spider(random_point(60))]
webs = []

next_fly_id = 1
elapsed_ms = 0  # defined before any fly is born, since birth_time_ms needs it

# --- Persistence: closing the window used to reset the whole lineage to zero.
# A save file (genome pool, generation/mutation counters, RNG seed, and the full
# lineage record) reloads on startup instead, so the population's history actually
# accumulates across sessions rather than restarting every time the game opens. ---
SAVE_PATH = "data/sim_state.json"
lineage = []  # every fly ever created: {fly_id, parent_ids, eye_hue, birth_ms, marked}


def make_fly(pos, genome=None, fly_id=None, birth_time_ms=None):
    """fly_id/birth_time_ms are only ever passed when restoring from a save file
    (see load_state below) — a brand-new fly always gets the next free id and is
    born 'now'. Every fly, restored or new, is recorded in the lineage list exactly
    once — keyed by fly_id, not by whether this is a fresh birth, because a fly
    restored from the mid-metamorphosis fallback pool (save_state's pending_genomes
    branch) never went through a normal make_fly() call in the session that made
    it, so it wouldn't have a lineage entry yet even though it isn't 'new'."""
    global next_fly_id
    genome = genome if genome is not None else random_genome()
    if fly_id is None:
        fly_id = next_fly_id
        next_fly_id += 1
    elif fly_id >= next_fly_id:
        next_fly_id = fly_id + 1
    if birth_time_ms is None:
        birth_time_ms = elapsed_ms
    if not any(entry["fly_id"] == fly_id for entry in lineage):
        lineage.append({
            "fly_id": fly_id, "parent_ids": genome.get("parent_ids"),
            "eye_hue": genome["eye_hue"], "birth_ms": birth_time_ms,
            "marked": genome.get("marked", False),
        })
    fly = Fly(pos, genome, fly_id, birth_time_ms)
    return fly


def make_starting_pair():
    return [make_fly([WORLD_W / 2 - 100, H / 2]), make_fly([WORLD_W / 2 + 100, H / 2])]


def save_state():
    """genome pool (just the living population — enough to reconstruct it), the
    generation/mutation counters, the RNG seed, and the full lineage record. Not
    positions or brain state: those reset to a fresh spawn each load, same as a
    real population doesn't remember exactly where it was standing."""
    if flies:
        pool = [
            {**{k: f.genome[k] for k in GENOME_TRAITS}, "fly_id": f.fly_id,
             "marked": f.marked, "parent_ids": f.genome.get("parent_ids"),
             "eye_hue_parents": f.genome.get("eye_hue_parents"),
             "birth_time_ms": f.birth_time_ms}
            for f in flies
        ]
    else:
        # Saved mid mating-merge or mid-maggot-growth, with no live Fly objects yet —
        # fall back to whatever genomes are queued in that animation so the population
        # is never actually wiped to zero just because of unlucky save timing. These
        # get a reserved id now and simply skip straight to being flies on load.
        pending_genomes = [mg.genome for mg in maggots]
        for m in metamorphoses:
            pending_genomes.extend(m.genome if m.mode == "merge_explode" else [m.genome])
        pool, reserved_id = [], next_fly_id
        for genome in pending_genomes:
            pool.append({**{k: genome[k] for k in GENOME_TRAITS}, "fly_id": reserved_id,
                         "marked": genome.get("marked", False), "parent_ids": genome.get("parent_ids"),
                         "eye_hue_parents": genome.get("eye_hue_parents"), "birth_time_ms": elapsed_ms})
            reserved_id += 1
    state = {
        "next_fly_id": next_fly_id,
        "generation_count": generation_count,
        "mutation_events": mutation_events,
        "rng_seed": rng_seed,
        "lineage": lineage,
        "genome_pool": pool,
    }
    try:
        tmp_path = SAVE_PATH + ".tmp"
        with open(tmp_path, "w") as fh:
            json.dump(state, fh)
        os.replace(tmp_path, SAVE_PATH)
    except OSError as e:
        print(f"[persistence] save failed: {e}")


def load_state():
    try:
        with open(SAVE_PATH) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        print(f"[persistence] save file unreadable ({e}) — starting fresh")
        return None


_saved = load_state()
if _saved is not None:
    next_fly_id = _saved["next_fly_id"]
    generation_count = _saved["generation_count"]
    mutation_events = _saved["mutation_events"]
    rng_seed = _saved["rng_seed"]
    lineage = _saved.get("lineage", [])
    random.seed(rng_seed)
    flies = []
    for g in _saved["genome_pool"]:
        genome = {k: g[k] for k in GENOME_TRAITS}
        genome["marked"] = g.get("marked", False)
        if g.get("parent_ids"):
            genome["parent_ids"] = tuple(g["parent_ids"])
        if g.get("eye_hue_parents"):
            genome["eye_hue_parents"] = tuple(g["eye_hue_parents"])
        spawn_pos = [WORLD_W / 2 + random.uniform(-150, 150), H / 2 + random.uniform(-100, 100)]
        flies.append(make_fly(spawn_pos, genome, fly_id=g["fly_id"], birth_time_ms=g.get("birth_time_ms", 0)))
    print(f"[persistence] restored {len(flies)} flies | generation {generation_count} | "
          f"{mutation_events} mutations | {len(lineage)} in lineage record | seed={rng_seed}")
else:
    rng_seed = random.randrange(2 ** 31)
    random.seed(rng_seed)
    flies = make_starting_pair()
    generation_count = 0
    print(f"[persistence] no save file found — starting fresh (seed={rng_seed})")

metamorphoses = []   # Metamorphosis objects mid-animation
maggots = []
last_save_ms = 0
SAVE_INTERVAL_MS = 30_000  # autosave every real 30s, in case the window never closes cleanly

# --- Mating season state ---
last_log_ms = 0
mating_season = False
heart_spawn_timer = 0
hearts = []
mating_snapshot = []    # both parents' brain panels, kept between the merge and the hatch
pending_offspring = 0   # >0 between a merge and the last hatch
next_mating_event_ms = random.uniform(MATING_SEASON_MIN_INTERVAL_MS, MATING_SEASON_MAX_INTERVAL_MS)
mating_over_flash = 0   # frames left on the "MATING'S OVER" banner
dance_angle = 0.0       # shared rotation angle for courting pairs
courtship_timers = {}   # frozenset({fly_id, fly_id}) -> ms spent circling close together

MATING_BUTTON_RECT = pygame.Rect(WORLD_W - 168, 10, 150, 30)
SIGN_SIZE = 190
MATING_SIGN_RECT = pygame.Rect(WORLD_W // 2 - SIGN_SIZE // 2, 60, SIGN_SIZE, SIGN_SIZE)

# --- Observation mode: hides the TRIGGER MATING button and the debug status lines
# (mutation/web counters, population/generation), leaving only the simulation, the
# brains, and the log — for whenever the manual mating trigger should read as a dev
# tool being folded away rather than a permanent fixture. Toggled with O. ---
observation_mode = False
# --- Title card: a one-paragraph artist statement, hidden by default, toggled
# with T — for whenever someone should encounter this cold. ---
title_card_visible = False

# --- Necropsy card: the marked fly (the one the frog always hunts) just vanishes
# on every other run. The instant it's actually eaten, flash one precise, cold
# card — exact genome, birth/death time, root IDs of whatever was mid-fire — then
# let it go forever. It is never written to the lineage record or anywhere else;
# this is the one acknowledgment that life got, and it doesn't repeat. ---
NECROPSY_DURATION_MS = 5000
NECROPSY_FADE_MS = 1200
necropsy_card = None


def build_necropsy_card(fly, death_ms):
    """Real genome values and real root IDs off the fly's actual spike activity at
    the moment it died — not synthesized after the fact."""
    idx = np.flatnonzero(fly.category_glow > GLOW_THRESHOLD)
    if idx.size:
        idx = idx[np.argsort(-fly.category_glow[idx])][:8]
        firing = [(int(fly.brain.ids[i]), float(fly.category_glow[i])) for i in idx]
    else:
        firing = []
    return {
        "fly_id": fly.fly_id,
        "genome": {k: fly.genome[k] for k in GENOME_TRAITS},
        "birth_ms": fly.birth_time_ms,
        "death_ms": death_ms,
        "firing": firing,
        "remaining_ms": NECROPSY_DURATION_MS,
    }


def start_mating_season():
    global mating_season, hearts, mating_over_flash
    if mating_season:
        return
    mating_season = True
    mating_over_flash = 0
    hearts = []


# ----------------------------------------------------------------------------
# Sonification: a very low, near-subliminal drone whose volume tracks total spike
# rate across both brains — the population literally humming. A single-cycle sine
# (plus a soft second harmonic) looped continuously; only its volume is touched
# per-frame, which is cheap and click-free since the buffer never changes.
# Entirely best-effort: a headless box or a machine with no audio device must
# never crash the sim over this — it just runs silent.
# ----------------------------------------------------------------------------
DRONE_HZ = 55
DRONE_BASE_VOLUME = 0.03
DRONE_GAIN = 3.5
DRONE_MAX_VOLUME = 0.20
drone_sound = None


def _init_drone():
    global drone_sound
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
        sample_rate = 44100
        n_samples = max(2, round(sample_rate / DRONE_HZ))  # integer cycle length -> click-free loop
        t = np.arange(n_samples) / sample_rate
        wave = np.sin(2 * np.pi * DRONE_HZ * t) * 0.7 + np.sin(2 * np.pi * DRONE_HZ * 2 * t) * 0.15
        samples = np.clip(wave * 32767, -32767, 32767).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack([samples, samples]))
        drone_sound = pygame.sndarray.make_sound(stereo)
        drone_sound.set_volume(DRONE_BASE_VOLUME)
        drone_sound.play(loops=-1)
    except Exception as e:
        print(f"[sonification] audio unavailable ({e}) — running silent")
        drone_sound = None


def update_drone(total_activity):
    if drone_sound is None:
        return
    try:
        drone_sound.set_volume(min(DRONE_MAX_VOLUME, DRONE_BASE_VOLUME + DRONE_GAIN * total_activity))
    except Exception:
        pass


_init_drone()


# ----------------------------------------------------------------------------
# Brain panel drawing
# ----------------------------------------------------------------------------
def draw_spike_trails(state, slot_x):
    """A thin line along a REAL synapse (from trail_targets, the top downstream
    partners by actual connectome weight), drawn only while both ends are
    actively firing hard (glow > TRAIL_HOT_THRESHOLD) — not merely lit. This is
    what makes activity visibly propagate hop-to-hop through the subgraph
    instead of just flickering independently at each site."""
    glow, cat, live = state["glow"], state["categories"], state["active"]
    xy, valid, targets = state["slot_xy"], state["slot_valid"], state["trail_targets"]
    hot = np.flatnonzero(valid & (glow > TRAIL_HOT_THRESHOLD))
    if not hot.size:
        return
    tgt_rows = targets[hot]
    for i, tgt_row in zip(hot.tolist(), tgt_rows.tolist()):
        c = cat[i]
        color = tuple((CATEGORY_PALETTE[c if live[c] else CAT_AMBIENT] * 0.5).astype(int).tolist())
        x0, y0 = slot_x + int(xy[i, 0]), int(xy[i, 1])
        for t in tgt_row:
            if t < 0 or not valid[t] or glow[t] <= TRAIL_HOT_THRESHOLD:
                continue
            x1, y1 = slot_x + int(xy[t, 0]), int(xy[t, 1])
            pygame.draw.line(screen, color, (x0, y0), (x1, y1), 1)


def draw_brain_dots(state, slot_x):
    """Draw every recently-active neuron, colored by its behavior only while that
    behavior is actually live (state["active"]) — the same level the FEAR/MATING
    readouts use — otherwise it reads as ambient. That applies to every neuron,
    including the hand-picked "core" sensory/command ones: a single incidental
    background spike in a real LC4 neuron shouldn't flash red when FEAR reads
    0.000. Core neurons are only drawn larger, not exempt from the gate.

    Brightness and radius also get a small pseudo-3D depth cue from the real
    soma_z (state["slot_depth"]) — the axis the flat x/y projection otherwise
    throws away — so a neuron toward the front of the brain reads as slightly
    nearer than one toward the back, instead of every dot looking equidistant."""
    glow, cat, core, live, depth = (state["glow"], state["categories"], state["core"],
                                    state["active"], state["slot_depth"])
    xy = state["slot_xy"]
    shown = np.flatnonzero(state["slot_valid"] & (glow > GLOW_THRESHOLD))
    if not shown.size:
        return

    draw_spike_trails(state, slot_x)

    cat_s, core_s, depth_s = cat[shown], core[shown], depth[shown]
    display_cat = np.where(live[cat_s], cat_s, CAT_AMBIENT)

    # Draw order (back to front): ambient non-core, ambient core, active-behavior
    # non-core, active-behavior core. Motion/vision are "core" too and fire almost
    # continuously — sorting on core-ness alone let a few hundred constantly-firing
    # ambient dots land on top of and visually bury a real, smaller, currently-active
    # cluster (e.g. the ~65 sugar neurons while eating). Whatever behavior is
    # actually live right now always wins the top layer, ambient or not.
    is_active = display_cat != CAT_AMBIENT
    priority = is_active.astype(np.int8) * 2 + core_s.astype(np.int8)
    order = np.argsort(priority, kind="stable")
    shown, display_cat, core_s, depth_s = shown[order], display_cat[order], core_s[order], depth_s[order]

    depth_factor = 0.7 + 0.3 * depth_s  # nearer (higher depth) pops slightly brighter/bigger

    # Ambient (motion/vision, always-on background firing) is deliberately pushed back —
    # dimmer AND smaller — so whatever behavior is actually live right now (sugar, fear,
    # reward...) reads as the clear foreground signal instead of competing on equal terms
    # with a few hundred constantly-firing background dots.
    is_active = display_cat != CAT_AMBIENT
    brightness_mult = np.where(is_active, 1.15, 0.5)
    brightness = np.minimum(1.0, (0.35 + 0.65 * np.minimum(1.0, glow[shown])) * depth_factor * brightness_mult)
    colors = np.clip(CATEGORY_PALETTE[display_cat] * brightness[:, None], 0, 255).astype(int)
    base_radius = np.where(is_active, np.where(core_s, 8, 5), np.where(core_s, 5, 2))
    radii = np.maximum(1, np.round(base_radius * (0.75 + 0.5 * depth_s))).astype(int)
    px = (slot_x + xy[shown, 0]).tolist()
    py = xy[shown, 1].tolist()
    for x, y, color, radius in zip(px, py, colors.tolist(), radii.tolist()):
        pygame.draw.circle(screen, color, (x, y), radius)


HOVER_RADIUS_PX = 8


def hover_lookup(state, slot_x, mouse_pos):
    """Nearest currently-lit neuron under the cursor, within HOVER_RADIUS_PX, or
    None. Only neurons actually being drawn are hoverable, so the tooltip only
    ever points at something the person can already see."""
    glow, xy, valid = state["glow"], state["slot_xy"], state["slot_valid"]
    shown = np.flatnonzero(valid & (glow > GLOW_THRESHOLD))
    if not shown.size:
        return None
    mx, my = mouse_pos
    px = slot_x + xy[shown, 0]
    py = xy[shown, 1]
    d2 = (px.astype(np.float32) - mx) ** 2 + (py.astype(np.float32) - my) ** 2
    j = int(np.argmin(d2))
    if d2[j] > HOVER_RADIUS_PX ** 2:
        return None
    idx = int(shown[j])
    return int(state["ids"][idx]), (int(px[j]), int(py[j])), int(state["categories"][idx])


def draw_hover_tooltip(root_id, pos, category_id):
    cell_type, cell_sub_class, super_class = neuron_meta(root_id)
    cat_label = CATEGORY_ID_TO_LABEL.get(category_id, "AMBIENT")
    lines = [
        (f"root_id {root_id}", (225, 225, 235)),
        (f"cell_type: {cell_type if cell_type != '?' else '(not annotated)'}", (200, 200, 215)),
        (f"sub_class: {cell_sub_class or '—'}", (200, 200, 215)),
        (f"super_class: {super_class or '—'}", (170, 170, 190)),
        (f"OWNER: {cat_label}", CATEGORY_COLORS.get(category_id, TEXT_LIGHT)),
    ]
    surfaces = [small_font.render(t, True, c) for t, c in lines]
    w = max(s.get_width() for s in surfaces) + 16
    h = sum(s.get_height() for s in surfaces) + 12
    x, y = pos[0] + 10, pos[1] + 10
    x = min(x, WORLD_W + PANEL_W - w - 4)
    y = min(y, H - h - 4)
    box = pygame.Surface((w, h), pygame.SRCALPHA)
    box.fill((12, 12, 18, 235))
    screen.blit(box, (x, y))
    pygame.draw.rect(screen, (90, 90, 110), (x, y, w, h), 1)
    ty = y + 6
    for s in surfaces:
        screen.blit(s, (x + 8, ty))
        ty += s.get_height()


def draw_feed_highlight(state, slot_x):
    """Big pulsing markers on the sensory neurons of the current eating/drinking event.
    Most gustatory events won't actually draw anything here (see note below) — the
    z-order and dimming in draw_brain_dots is what makes those events visible instead."""
    kind = state["feed_event"]
    if kind not in FEED_EVENT_COLORS or state["feed_remaining_ms"] <= 0:
        return
    idx = np.asarray(state["feed_indices"].get(kind, []), dtype=int)
    idx = idx[state["slot_valid"][idx]] if len(idx) else idx
    pulse = 0.72 + 0.28 * abs(np.sin(pygame.time.get_ticks() / 115.0))
    color = FEED_EVENT_COLORS[kind]
    halo = tuple(int(c * 0.45) for c in color)
    core = tuple(min(255, int(c * (0.95 + 0.05 * pulse))) for c in color)
    # The actual sensory neurons (idx) rarely have a real soma position to draw at —
    # every gustatory (sugar/water/salt/bitter) STIMULI neuron lacks one entirely,
    # since taste receptor neurons sit in the proboscis, outside the imaged soma
    # volume. When they do have one (other categories), draw the per-neuron markers;
    # this is real anatomy, not a bug, so it's a normal branch, not an error case.
    for i in idx:
        center = (int(slot_x + state["slot_xy"][i, 0]), int(state["slot_xy"][i, 1]))
        pygame.draw.circle(screen, (0, 0, 0), center, 13)
        pygame.draw.circle(screen, halo, center, int(9 + 3 * pulse))
        pygame.draw.circle(screen, core, center, 8)
        pygame.draw.circle(screen, (255, 255, 255), center, 2)



def draw_panel_text(state, slot_x):
    """Single column now — each fly's sub-panel is narrower than the old full-width
    stacked layout, so a two-column readout no longer fits and isn't needed: there's
    plenty of vertical room now that each brain gets the FULL window height instead
    of half of it."""
    panel_x = slot_x + 24
    left, right, fear, mating = state["left"], state["right"], state["fear"], state["mating"]
    reward, punish = state["reward"], state["punish"]
    smell_left, smell_right = state["smell_left"], state["smell_right"]
    desc_left, desc_right = state["desc_left"], state["desc_right"]

    if state["snapshot"]:
        header = f"FLY {state['fly_id']:02d} — MATING SNAPSHOT"
        header_color = PINK
        subtitle = "FULL FLYWIRE BRAIN  |  pC1 MATING SIGNAL"
    else:
        header = f"FLY {state['fly_id']:02d} — FULL NEURAL ACTIVITY"
        header_color = PINK if mating > 0.03 else (230, 230, 230)
        subtitle = "REAL FLYWIRE NEURONS  |  pC1 MATING SIGNAL ACTIVE" if mating > 0.03 else "REAL FLYWIRE NEURONS"

    screen.blit(panel_bold.render(header, True, header_color), (panel_x, 16))
    screen.blit(small_font.render(subtitle, True, (150, 150, 170)), (panel_x, 44))
    pygame.draw.line(screen, (70, 70, 90), (panel_x, 68), (slot_x + SLOT_W - 20, 68))

    y = 80
    rows = [
        ("VISUAL LEFT", f"{left:0.3f}", TEXT_LIGHT),
        ("VISUAL RIGHT", f"{right:0.3f}", TEXT_LIGHT),
        ("SMELL L / R", f"{smell_left:0.3f} / {smell_right:0.3f}", TEXT_LIGHT),
        ("FEAR", f"{fear:0.3f}", (255, 170, 90) if fear > 0.03 else TEXT_LIGHT),
        ("MATING", f"{mating:0.3f}", PINK if mating > 0.03 else TEXT_LIGHT),
        ("REWARD (PAM)", f"{reward:0.3f}", CATEGORY_COLORS[CAT_REWARD] if reward > 0.03 else TEXT_LIGHT),
        ("PUNISH (PPL1)", f"{punish:0.3f}", CATEGORY_COLORS[CAT_PUNISH] if punish > 0.03 else TEXT_LIGHT),
    ]
    for name, value, color in rows:
        screen.blit(panel_font.render(f"{name:<16} {value}", True, color), (panel_x, y))
        y += 26

    y += 8
    # The real output readout — this, not vision, is what TURN BIAS is decoded from.
    desc_color = (120, 220, 255) if abs(desc_right - desc_left) > 0.01 else TEXT_LIGHT
    screen.blit(panel_font.render(f"{'DESCENDING (DN)':<16} {desc_left:0.3f} / {desc_right:0.3f}", True, desc_color),
                (panel_x, y)); y += 26
    turn_bias = desc_right - desc_left
    direction = "RIGHT" if turn_bias > 0.001 else "LEFT" if turn_bias < -0.001 else "STRAIGHT"
    screen.blit(panel_font.render(f"TURN BIAS        {direction}", True, TEXT_LIGHT), (panel_x, y)); y += 26
    screen.blit(panel_font.render(f"pC1a/b/c         {mating:0.3f}", True, PINK), (panel_x, y)); y += 32

    # Dopamine is a modulator, not a state of its own — surfaced as a small badge
    # next to the main STATE line rather than replacing it, so "EATING SUGAR" and
    # "REWARD FIRING" can both be true and both visible at once.
    dopamine_note = None
    if punish > 0.03:
        dopamine_note = ("PPL1 PUNISH FIRING", CATEGORY_COLORS[CAT_PUNISH])
    elif reward > 0.03:
        dopamine_note = ("PAM REWARD FIRING", CATEGORY_COLORS[CAT_REWARD])

    bar_w = SLOT_W - 68
    kind, remaining = state["feed_event"], state["feed_remaining_ms"]
    if kind in FEED_EVENT_LABELS and remaining > 0:
        color = FEED_EVENT_COLORS[kind]
        verb = "EATING" if kind in ("sugar", "salt") else "DRINKING"
        screen.blit(small_font.render(f"STATE: {verb} ({kind.upper()})", True, color), (panel_x, y)); y += 26
        screen.blit(panel_font.render(f"{FEED_EVENT_LABELS[kind]}  {remaining / 1000.0:0.1f}s", True, color),
                    (panel_x, y)); y += 28
        frac = max(0.0, min(1.0, remaining / FEED_EVENT_DURATION_MS))
        pygame.draw.rect(screen, (45, 45, 55), (panel_x, y, bar_w, 10), border_radius=4)
        pygame.draw.rect(screen, color, (panel_x, y, int(bar_w * frac), 10), border_radius=4)
        y += 22
        screen.blit(panel_font.render(f"EVENT NEURONS {state['feed_neurons']}   SPIKES {state['feed_spikes']}",
                                      True, TEXT_LIGHT), (panel_x, y)); y += 26
        if dopamine_note:
            screen.blit(small_font.render(dopamine_note[0], True, dopamine_note[1]), (panel_x, y))
    else:
        status = "MATING / COURTSHIP" if mating > 0.03 else ("FEAR RESPONSE" if fear > 0.03 else "EXPLORING")
        screen.blit(small_font.render(f"STATE: {status}", True, (160, 160, 175)), (panel_x, y)); y += 26
        if dopamine_note:
            screen.blit(small_font.render(dopamine_note[0], True, dopamine_note[1]), (panel_x, y))


def draw_brain_slot(slot, state, mouse_pos=None):
    slot_x = WORLD_W + slot * SLOT_W
    screen.blit(brain_bg_half, (slot_x, 0))
    if state is None:
        screen.blit(panel_bold.render("NO ACTIVE FLY", True, (180, 180, 190)), (slot_x + 18, 18))
        return
    draw_brain_dots(state, slot_x)
    draw_feed_highlight(state, slot_x)
    draw_panel_text(state, slot_x)
    if mouse_pos is not None and slot_x <= mouse_pos[0] < slot_x + SLOT_W and 0 <= mouse_pos[1] < H:
        hit = hover_lookup(state, slot_x, mouse_pos)
        if hit is not None:
            root_id, dot_pos, category_id = hit
            draw_hover_tooltip(root_id, dot_pos, category_id)


def draw_legend():
    x, y = WORLD_W + 14, H - 15
    for label, cat in LEGEND:
        pygame.draw.circle(screen, CATEGORY_COLORS[cat], (x + 4, y + 6), 4)
        text = small_font.render(label, True, (170, 170, 185))
        screen.blit(text, (x + 12, y))
        x += 12 + text.get_width() + 12


def draw_necropsy_card(card):
    """A single, cold acknowledgment — real genome, real birth/death timestamps,
    real root IDs off whatever was actually mid-fire — flashed once over the
    world view and then gone. Nothing here is stored anywhere else."""
    fade = 1.0
    if card["remaining_ms"] < NECROPSY_FADE_MS:
        fade = max(0.0, card["remaining_ms"] / NECROPSY_FADE_MS)
    alpha = int(235 * fade)
    if alpha <= 0:
        return

    w, h = 480, 300
    x0, y0 = (WORLD_W - w) // 2, (H - h) // 2
    card_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    card_surf.fill((8, 8, 10, alpha))
    pygame.draw.rect(card_surf, (200, 40, 40, alpha), (0, 0, w, h), 2)
    screen.blit(card_surf, (x0, y0))

    def line(text, y, font=panel_font, color=(210, 210, 215)):
        surf = font.render(text, True, color)
        surf.set_alpha(alpha)
        screen.blit(surf, (x0 + 24, y))

    line("NECROPSY", y0 + 18, font=panel_bold, color=(220, 60, 60))
    line(f"SPECIMEN FLY {card['fly_id']:02d}", y0 + 48, font=small_font, color=(170, 170, 180))
    y = y0 + 80
    g = card["genome"]
    line(f"EYE HUE          {g['eye_hue']:0.1f}°", y); y += 24
    line(f"TURN GAIN        {g['turn_gain']:0.4f}", y); y += 24
    line(f"SIGHT RADIUS     {g['sight_radius']:0.1f}", y); y += 24
    line(f"BASE SPEED       {g['base_speed']:0.3f}", y); y += 24
    line(f"TIME OF BIRTH    {card['birth_ms'] / 1000.0:0.1f}s", y); y += 24
    line(f"TIME OF DEATH    {card['death_ms'] / 1000.0:0.1f}s", y); y += 32
    if card["firing"]:
        line("MID-FIRE AT DEATH:", y, font=small_font, color=(170, 170, 180)); y += 20
        for root_id, glow in card["firing"][:5]:
            display_type, _ = neuron_display_type(root_id)
            line(f"  {root_id}  {display_type}", y, font=mono_font, color=(190, 190, 200)); y += 18
    else:
        line("MID-FIRE AT DEATH:  none", y, font=small_font, color=(170, 170, 180))


# --- Title card: one paragraph of framing, hidden by default, toggled with T —
# for whenever someone should encounter this cold instead of mid-simulation. ---
TITLE_CARD_TEXT = (
    "A real fruit fly connectome, extracted, mapped, and set to run. Two specimens "
    "forage, court, and are hunted, driven entirely by real neurons and their real "
    "synaptic weights — not a model of behavior, the wiring itself, doing what it "
    "was built to do. There is no objective here, no score, and no way to win. The "
    "population persists only because nothing stops it."
)


def draw_title_card():
    overlay_w, overlay_h = WORLD_W + PANEL_W, H + LOG_H
    overlay = pygame.Surface((overlay_w, overlay_h), pygame.SRCALPHA)
    overlay.fill((6, 6, 8, 246))
    screen.blit(overlay, (0, 0))

    title = event_font.render("FLY BRAIN", True, (225, 225, 230))
    screen.blit(title, title.get_rect(center=(overlay_w // 2, overlay_h // 2 - 96)))

    y = overlay_h // 2 - 30
    for line_text in textwrap.wrap(TITLE_CARD_TEXT, width=64):
        surf = panel_font.render(line_text, True, (185, 185, 195))
        screen.blit(surf, surf.get_rect(center=(overlay_w // 2, y)))
        y += 28

    hint = small_font.render("T  close", True, (110, 110, 125))
    screen.blit(hint, hint.get_rect(center=(overlay_w // 2, overlay_h - 40)))


# ----------------------------------------------------------------------------
# Clinical neural event log: a scrolling monospace strip of real firing neurons
# (root_id, cell_type, owning behavior), sampled off actual spike activity —
# never synthesized — so it reads like an instrument log, not a placeholder.
# ----------------------------------------------------------------------------
LOG_SAMPLE_INTERVAL_MS = 220
LOG_MAX_LINES = 8
event_log = deque(maxlen=LOG_MAX_LINES)
mono_font = pygame.font.SysFont("couriernew,consolas,monospace", 16)
mono_header_font = pygame.font.SysFont("couriernew,consolas,monospace", 16, bold=True)


def sample_event_log(flies, elapsed_ms):
    for fly in flies:
        top = fly.top_active_neuron()
        if top is None:
            continue
        root_id, cat_id, glow_val = top
        _, sub_class, _ = neuron_meta(root_id)
        display_type, is_fallback = neuron_display_type(root_id)
        if is_fallback:
            display_type = f"[{display_type}]"  # bracketed = cell_type was blank; this is the coarser class/super_class
        event_log.append({
            "t_ms": elapsed_ms, "fly_id": fly.fly_id, "root_id": root_id,
            "cell_type": display_type, "sub_class": sub_class,
            "label": CATEGORY_ID_TO_LABEL.get(cat_id, "AMBIENT"),
            "color": CATEGORY_COLORS.get(cat_id, TEXT_LIGHT), "glow": glow_val,
        })


def draw_event_log():
    x0, y0 = 10, H + 8
    w = WORLD_W + PANEL_W - 20
    box = pygame.Surface((w, LOG_H - 16), pygame.SRCALPHA)
    box.fill((8, 8, 12, 255))
    screen.blit(box, (x0, y0))
    pygame.draw.rect(screen, (55, 55, 70), (x0, y0, w, LOG_H - 16), 1)

    header = mono_header_font.render(
        f"{'TIME':>8}  {'FLY':<4}{'ROOT_ID':>19}  {'CELL_TYPE':<14}{'SUB_CLASS':<12}{'CIRCUIT':<9}",
        True, (150, 150, 170))
    screen.blit(header, (x0 + 8, y0 + 4))
    pygame.draw.line(screen, (55, 55, 70), (x0 + 6, y0 + 24), (x0 + w - 6, y0 + 24))

    for i, entry in enumerate(reversed(event_log)):
        t_s = entry["t_ms"] / 1000.0
        line = (f"{t_s:8.2f}  {entry['fly_id']:02d}  {entry['root_id']:>18}  "
               f"{entry['cell_type']:<14}{entry['sub_class'] or '-':<12}{entry['label']:<9}")
        fade = max(0.45, 1.0 - i * 0.07)
        color = tuple(int(c * fade) for c in entry["color"])
        screen.blit(mono_font.render(line, True, color), (x0 + 8, y0 + 30 + i * 15))


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not observation_mode and MATING_BUTTON_RECT.collidepoint(event.pos):
                start_mating_season()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_o:
                observation_mode = not observation_mode
            elif event.key == pygame.K_t:
                title_card_visible = not title_card_visible

    dt_ms = clock.get_time() or 1000 / FPS
    elapsed_ms += dt_ms

    # Mating season is a random event, re-rolled 5-10 real hours after the last one ends.
    if not mating_season and elapsed_ms >= next_mating_event_ms:
        start_mating_season()

    dance_angle += DANCE_ANGULAR_SPEED

    # --- predators ---
    frog.update(flies, dt_ms, mating_season)
    if frog.ate is not None:
        if frog.ate.marked:
            necropsy_card = build_necropsy_card(frog.ate, elapsed_ms)
        flies = [f for f in flies if f is not frog.ate]
        frog.ate = None

    if necropsy_card is not None:
        necropsy_card["remaining_ms"] -= dt_ms
        if necropsy_card["remaining_ms"] <= 0:
            necropsy_card = None
    for spider in spiders:
        spider.update(flies, webs, dt_ms, mating_season)
    for web in webs:
        web.update(dt_ms)
    webs = [w for w in webs if not w.finished]
    flies = [f for f in flies if not (f.doomed and f.webbed_ms <= 0)]

    # --- flies ---
    for fly in flies:
        fly.update(fruits, frog, waters, salts, spiders, flies, mating_season, dance_angle, dt_ms)

    update_drone(sum(f.total_activity for f in flies) if flies else 0.0)

    if elapsed_ms - last_log_ms >= LOG_SAMPLE_INTERVAL_MS:
        last_log_ms = elapsed_ms
        sample_event_log(flies, elapsed_ms)

    # --- courtship: a pair circling close together long enough produces offspring ---
    to_remove = set()
    active_pair_keys = set()
    if mating_season and pending_offspring == 0:
        for i in range(len(flies)):
            for j in range(i + 1, len(flies)):
                if i in to_remove or j in to_remove:
                    continue
                if flies[i].mating_cooldown > 0 or flies[j].mating_cooldown > 0:
                    continue
                dist = np.linalg.norm(flies[i].pos - flies[j].pos)
                if not (COURTSHIP_TRIGGER_MIN_DIST <= dist <= COURTSHIP_TRIGGER_MAX_DIST):
                    continue
                key = frozenset((flies[i].fly_id, flies[j].fly_id))
                active_pair_keys.add(key)
                courtship_timers[key] = courtship_timers.get(key, 0) + dt_ms
                if courtship_timers[key] < COURTSHIP_DURATION_MS:
                    continue

                # Three offspring, each an independent mutation roll; the third is marked
                # (hunted by the frog, not counted in the generation total).
                child_genomes = [mutate_genome(flies[i].genome, flies[j].genome) for _ in range(3)]
                for child in child_genomes:
                    child["parent_ids"] = (flies[i].fly_id, flies[j].fly_id)
                child_genomes[2]["marked"] = True
                metamorphoses.append(Metamorphosis((flies[i].pos + flies[j].pos) / 2, child_genomes, "merge_explode"))

                # Keep both parents' brain panels visible until the offspring hatch.
                parents = sorted((flies[i], flies[j]), key=lambda f: f.fly_id)
                mating_snapshot = [{**p.panel_state(), "snapshot": True} for p in parents]

                for h in hearts:
                    h.damaged = True
                pending_offspring += 3
                to_remove.update((i, j))
    courtship_timers = {k: v for k, v in courtship_timers.items() if k in active_pair_keys}
    flies = [f for idx, f in enumerate(flies) if idx not in to_remove]

    # --- metamorphosis ---
    for m in metamorphoses:
        m.update()
    finished_merges = [m for m in metamorphoses if m.finished and m.mode == "merge_explode"]
    finished_hatches = [m for m in metamorphoses if m.finished and m.mode == "hatch"]
    metamorphoses = [m for m in metamorphoses if not m.finished]

    for m in finished_merges:
        for idx, genome in enumerate(m.genome):
            offset = np.array([random.uniform(-6, 6), random.uniform(-6, 6)])
            duration = 600 + idx * random.randint(20, 60)  # stagger so they don't hatch in lockstep
            maggots.append(Maggot(m.pos + offset, genome, duration=duration))

    for m in finished_hatches:
        new_fly = make_fly(m.pos, m.genome)
        flies.append(new_fly)
        if not new_fly.marked:
            generation_count += 1
        pending_offspring = max(0, pending_offspring - 1)
        if pending_offspring == 0:
            mating_snapshot = []
            if mating_season:
                # All offspring born: mating season ends and the next one is re-rolled.
                mating_season = False
                hearts = []
                mating_over_flash = 300
                next_mating_event_ms = elapsed_ms + random.uniform(MATING_SEASON_MIN_INTERVAL_MS,
                                                                   MATING_SEASON_MAX_INTERVAL_MS)

    # --- maggots hatch when ready ---
    for mg in maggots:
        mg.update()
    for mg in [mg for mg in maggots if mg.timer <= 0]:
        metamorphoses.append(Metamorphosis(mg.pos, mg.genome, "hatch"))
    maggots = [mg for mg in maggots if mg.timer > 0]

    # --- hearts pop up around the closest courting pair ---
    closest_pair_mid, closest_pair_dist = None, None
    if mating_season:
        for i in range(len(flies)):
            for j in range(i + 1, len(flies)):
                d = np.linalg.norm(flies[i].pos - flies[j].pos)
                if d <= ORBIT_ENTER_DIST and (closest_pair_dist is None or d < closest_pair_dist):
                    closest_pair_dist = d
                    closest_pair_mid = (flies[i].pos + flies[j].pos) / 2

    if mating_season and pending_offspring == 0 and closest_pair_mid is not None:
        heart_spawn_timer += 1
        if heart_spawn_timer >= MATING_HEART_SPAWN_INTERVAL:
            heart_spawn_timer = 0
            love_radius = 30
            hearts.append(Heart((closest_pair_mid[0] + random.uniform(-love_radius, love_radius),
                                 closest_pair_mid[1] + random.uniform(-love_radius, love_radius))))

    for heart in hearts:
        heart.update(dt_ms)
    hearts = [h for h in hearts if not h.finished][-45:]

    # never let the piece fully stop
    if not flies and not maggots and not metamorphoses:
        flies = make_starting_pair()

    # --- draw world ---
    screen.blit(world_bg, (0, 0))
    screen.blit(darken_overlay, (0, 0))
    for fruit in fruits:
        screen.blit(fruit["sprite"], fruit["sprite"].get_rect(center=fruit["pos"].astype(int)))
    for w in waters:
        screen.blit(water_sprite, water_sprite.get_rect(center=w["pos"].astype(int)))
    for s in salts:
        screen.blit(s["sprite"], s["sprite"].get_rect(center=s["pos"].astype(int)))
    frog.draw(screen)
    for spider in spiders:
        spider.draw(screen)
    for web in webs:
        web.draw(screen)
    for mg in maggots:
        mg.draw(screen)
    for m in metamorphoses:
        m.draw(screen)

    if mating_season:
        sign_surf = pygame.Surface(MATING_SIGN_RECT.size, pygame.SRCALPHA)
        sign_surf.fill((18, 10, 16, 215))
        screen.blit(sign_surf, MATING_SIGN_RECT.topleft)
        pygame.draw.rect(screen, PINK, MATING_SIGN_RECT, 3)
        pygame.draw.rect(screen, PINK, MATING_SIGN_RECT.inflate(-14, -14), 1)
        for line_i, line in enumerate(("MATING", "SEASON")):
            txt = event_font.render(line, True, PINK)
            screen.blit(txt, txt.get_rect(center=(MATING_SIGN_RECT.centerx,
                                                  MATING_SIGN_RECT.centery - 22 + line_i * 44)))

    for heart in hearts:
        heart.draw(screen)

    for fly in flies:
        fly.draw(screen)
        label = small_font.render(f"FLY {fly.fly_id:02d}", True, (235, 235, 235))
        screen.blit(label, label.get_rect(center=(int(fly.pos[0]), int(fly.pos[1] - FLY_DISPLAY_SIZE * 0.62))))

    if mating_over_flash > 0:
        mating_over_flash -= 1
        banner_color = (150, 104, 122)
        over = event_font.render("MATING'S OVER", True, banner_color)
        over = pygame.transform.scale(over, (int(over.get_width() * 1.6), int(over.get_height() * 1.6)))
        over_rect = over.get_rect(center=(WORLD_W // 2, H // 2))
        box = over_rect.inflate(46, 30)
        box_surf = pygame.Surface(box.size, pygame.SRCALPHA)
        box_surf.fill((14, 10, 13, 225))
        screen.blit(box_surf, box.topleft)
        pygame.draw.rect(screen, banner_color, box, 2)
        if mating_over_flash < 60:  # fade out over the final second
            over.set_alpha(int(255 * mating_over_flash / 60))
        screen.blit(over, over_rect)

    # --- status lines and mating button (folded away in observation mode) ---
    if not observation_mode:
        screen.blit(small_font.render(
            f"MUTATIONS: {mutation_events}  |  WEB SHOTS: {predator_stats['web_shots']}  |  "
            f"WEB HITS: {predator_stats['web_hits']}", True, (190, 190, 200)), (10, H - 52))
        screen.blit(font.render(
            f"POPULATION {len(flies):02d}   MAGGOTS {len(maggots):02d}   GENERATION {generation_count:03d}",
            True, (255, 255, 255)), (10, H - 28))

        pygame.draw.rect(screen, (70, 20, 40) if mating_season else (150, 30, 80), MATING_BUTTON_RECT,
                          border_radius=6)
        pygame.draw.rect(screen, PINK, MATING_BUTTON_RECT, 2, border_radius=6)
        btn_text = small_font.render("MATING ACTIVE" if mating_season else "TRIGGER MATING", True, (255, 255, 255))
        screen.blit(btn_text, btn_text.get_rect(center=MATING_BUTTON_RECT.center))

        hint = small_font.render("O  observation mode    T  title card", True, (90, 90, 105))
        screen.blit(hint, (WORLD_W - hint.get_width() - 14, 46))

    # --- neural panel: one full brain per fly (or the frozen parents during a mating) ---
    mouse_pos = pygame.mouse.get_pos()
    live_states = [f.panel_state() for f in sorted(flies, key=lambda f: f.fly_id)[:2]]
    for slot in range(2):
        if slot < len(live_states):
            state = live_states[slot]
        elif slot < len(mating_snapshot):
            state = mating_snapshot[slot]
        else:
            state = None
        draw_brain_slot(slot, state, mouse_pos)
    pygame.draw.line(screen, (60, 60, 80), (WORLD_W + SLOT_W, 0), (WORLD_W + SLOT_W, H), 1)
    draw_legend()
    draw_event_log()
    if necropsy_card is not None:
        draw_necropsy_card(necropsy_card)
    if title_card_visible:
        draw_title_card()

    pygame.display.flip()
    clock.tick(FPS)

    if elapsed_ms - last_save_ms >= SAVE_INTERVAL_MS:
        last_save_ms = elapsed_ms
        save_state()

save_state()
pygame.quit()
