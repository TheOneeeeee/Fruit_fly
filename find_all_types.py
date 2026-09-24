import pandas as pd
ann = pd.read_csv("data/flywire_annotations.tsv", sep="\t", low_memory=False)

def get_type(pattern):
    matches = ann[ann["cell_type"].str.contains(pattern, na=False, regex=False)]
    return matches["root_id"].tolist(), matches["cell_type"].unique()

for name, pattern in [("LC4 (looming)", "LC4"), ("LPLC2 (looming)", "LPLC2"),
                       ("PAM (dopamine/reward)", "PAM"), ("PPL1 (dopamine/punishment)", "PPL1"),
                       ("pC1 (mating, female)", "pC1")]:
    ids, types_found = get_type(pattern)
    print(f"{name}: {len(ids)} neurons, types matched: {list(types_found)[:5]}")