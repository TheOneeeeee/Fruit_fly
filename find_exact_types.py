import pandas as pd
ann = pd.read_csv("data/flywire_annotations.tsv", sep="\t", low_memory=False)

def get_exact(*type_names):
    matches = ann[ann["cell_type"].isin(type_names)]
    return matches["root_id"].tolist()

groups = {
    "LC4": get_exact("LC4"),
    "LPLC2": get_exact("LPLC2"),
    "PAM": get_exact(*[t for t in ann["cell_type"].dropna().unique() if str(t).startswith("PAM")]),
    "PPL1": get_exact(*[t for t in ann["cell_type"].dropna().unique() if str(t).startswith("PPL1")]),
    "pC1_mating": get_exact("pC1a", "pC1b", "pC1c"),
    "pC1_aggression": get_exact("pC1d", "pC1e"),
}
for name, ids in groups.items():
    print(f"{name}: {len(ids)} neurons")