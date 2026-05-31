"""
Corre esto UNA VEZ para ver la estructura real de los datasets.
Resultado: pega la salida en el chat.
    python inspect_datasets.py
"""
import kagglehub, pandas as pd
from pathlib import Path

print("=" * 60)
print("DATASET 1 — Travel Recommendation")
print("=" * 60)
p1 = Path(kagglehub.dataset_download("amanmehra23/travel-recommendation-dataset"))
for csv in sorted(p1.rglob("*.csv")):
    df = pd.read_csv(csv, nrows=3)
    print(f"\n{csv.name}  ({df.shape[1]} cols)")
    print("  Columnas:", list(df.columns))
    for col in df.columns:
        print(f"    {col}: {df[col].iloc[0]!r}")

print("\n" + "=" * 60)
print("DATASET 2 — Driver Behavior Images")
print("=" * 60)
p2 = Path(kagglehub.dataset_download("arafatsahinafridi/multi-class-driver-behavior-image-dataset"))
for item in sorted(p2.rglob("*")):
    if item.is_dir():
        imgs = list(item.glob("*.jpg")) + list(item.glob("*.png")) + list(item.glob("*.jpeg"))
        if imgs:
            print(f"  CLASS FOLDER: {item.relative_to(p2)}  ({len(imgs)} imgs)")
