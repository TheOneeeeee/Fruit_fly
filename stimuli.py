import pandas as pd

_ann = pd.read_csv("data/flywire_annotations.tsv", sep="\t", low_memory=False)

def _exact(*type_names):
    return _ann[_ann["cell_type"].isin(type_names)]["root_id"].tolist()

def _startswith(prefix):
    types = [t for t in _ann["cell_type"].dropna().unique() if str(t).startswith(prefix)]
    return _exact(*types)

def _sub_class(cell_class, *sub_classes):
    # cell_type in this dataset is just anatomical naming (LB3, PhG1a, ...) —
    # it doesn't say what a gustatory neuron actually responds to. cell_sub_class
    # is the real modality label (e.g. "sugar/water", "bitter", "low-salt").
    rows = _ann[(_ann["cell_class"] == cell_class) & (_ann["cell_sub_class"].isin(sub_classes))]
    return rows["root_id"].tolist()

_mechano = _ann[_ann["cell_class"] == "mechanosensory"].sample(n=200, random_state=42)["root_id"].tolist()
_visual = _ann[_ann["super_class"].isin(["optic", "visual_projection"])]
_vision_left = _visual[_visual["side"] == "left"].sample(n=150, random_state=42)["root_id"].tolist()
_vision_right = _visual[_visual["side"] == "right"].sample(n=150, random_state=42)["root_id"].tolist()

# Olfactory receptor neurons (ORNs) — real smell, distinct from the pheromone-tuned
# subset of the same cell_class (those are courtship-relevant, not food-relevant, so
# excluded here to keep this population read as "food odor"). 2282 total in the
# dataset; sampled down like vision/motion for sim cost, split left/right antenna so
# it can drive a genuine left/right chemotaxis bias the same way vision does — but
# unlike vision, the game drives it by distance only (no sight_radius cutoff, no
# angle-of-view cone), since smell isn't blocked by walls or limited to what's ahead.
_olfactory = _ann[(_ann["cell_class"] == "olfactory") & (_ann["cell_sub_class"].isna())]
_olfactory_left = _olfactory[_olfactory["side"] == "left"].sample(n=150, random_state=42)["root_id"].tolist()
_olfactory_right = _olfactory[_olfactory["side"] == "right"].sample(n=150, random_state=42)["root_id"].tolist()

# There's no biologically separate "water-only" GRN type in this dataset —
# sugar and water are both driven by the same 129-neuron "sugar/water"
# sub_class (mostly LB3 + LB2d). We split that pool deterministically in
# half rather than reusing the same neurons for both, so the brain panel
# shows sugar and water as genuinely distinct (if overlapping-origin)
# populations instead of two labels pointed at identical neurons.
_sugar_water = sorted(_sub_class("gustatory", "sugar/water"))
_sugar_ids = _sugar_water[0::2]
_water_ids = _sugar_water[1::2]

STIMULI = {
    "sugar": {"ids": _sugar_ids, "hz": 200},
    # Full bitter class (65 neurons: LB1e/LB1c/LB1a/LB1d/LB1b), up from a
    # partial hand-picked 42 that also secretly leaked 11 of the old "salt"
    # IDs — see the salt fix below.
    "bitter": {"ids": _sub_class("gustatory", "bitter"), "hz": 200},
    "fear": {"ids": _exact("LC4") + _exact("LPLC2"), "hz": 150},
    "dopamine_reward": {"ids": _startswith("PAM"), "hz": 100},
    "dopamine_punish": {"ids": _startswith("PPL1"), "hz": 100},
    "mating": {"ids": _exact("pC1a", "pC1b", "pC1c"), "hz": 150},
    "aggression": {"ids": _exact("pC1d", "pC1e"), "hz": 150},
    "motion": {"ids": _mechano, "hz": 30},
    "vision_left": {"ids": _vision_left, "hz": 50},
    "vision_right": {"ids": _vision_right, "hz": 50},
    "water": {"ids": _water_ids, "hz": 200},
    # Was mislabeled: 11 of the old 18 "salt" IDs were actually bitter-class
    # neurons. Real low-salt class is only 19 neurons (LB2a-b/LB2c/LB4a) —
    # we now use all of them instead of a hand-picked, partly-wrong subset.
    "salt": {"ids": _sub_class("gustatory", "low-salt"), "hz": 200},
    "olfactory_left": {"ids": _olfactory_left, "hz": 120},
    "olfactory_right": {"ids": _olfactory_right, "hz": 120},
}

if __name__ == "__main__":
    for name, cfg in STIMULI.items():
        print(f"{name}: {len(cfg['ids'])} neurons @ {cfg['hz']}Hz")
