import pandas as pd
ann = pd.read_csv("data/flywire_annotations.tsv", sep="\t")
print(ann.columns.tolist())
print(ann.head())