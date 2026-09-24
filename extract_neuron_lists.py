import pandas as pd
ann = pd.read_csv("data/flywire_annotations.tsv", sep="\t", low_memory=False)
print("=== cell_class ===")
print(ann["cell_class"].value_counts().head(30))
print("\n=== super_class ===")
print(ann["super_class"].value_counts())