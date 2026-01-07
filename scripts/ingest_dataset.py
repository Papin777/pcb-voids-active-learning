import shutil
from pathlib import Path

SRC = Path("data/raw/True_Labelisation.v1i")
OUT_IMG = Path("data/processed/images")
OUT_LBL = Path("data/processed/labels")

# création des dossiers cibles
for sp in ("train", "val"):
    (OUT_IMG / sp).mkdir(parents=True, exist_ok=True)
    (OUT_LBL / sp).mkdir(parents=True, exist_ok=True)

def copy_split(img_dir: Path, lbl_dir: Path, dst: str):
    imgs = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for img in imgs:
        shutil.copy2(img, OUT_IMG / dst / img.name)
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, OUT_LBL / dst / lbl.name)
        else:
            # crée un .txt vide si pas d'étiquette
            (OUT_LBL / dst / img.with_suffix(".txt").name).write_text("", encoding="utf-8")

# copies Roboflow (train/ valid/ test/)
val_name = "valid" if (SRC / "valid").exists() else ("val" if (SRC / "val").exists() else None)
copy_split(SRC / "train" / "images", SRC / "train" / "labels", "train")
if val_name:
    copy_split(SRC / val_name / "images", SRC / val_name / "labels", "val")

# crée le fichier data.yaml avec les bons noms de classes
yaml_text = """path: .
train: images/train
val: images/val
names:
  0: chip
  1: void
"""
Path("data/processed/data.yaml").write_text(yaml_text, encoding="utf-8")

print("✓ Ingestion terminée → data/processed/")
