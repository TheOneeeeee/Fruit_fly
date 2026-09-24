import pandas as pd
ann = pd.read_csv("data/flywire_annotations.tsv", sep="\t", low_memory=False)
print(ann["side"].value_counts())
print()
visual = ann[ann["super_class"].isin(["optic", "visual_projection"])]
print(visual["side"].value_counts())