import json

with open("code/paper-phil-drosophila/figures.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    if ("neu_water" in source or "neu_ir94e" in source) and "=" in source and "[" in source:
        print("--- CELL ---")
        print(source)
        print()