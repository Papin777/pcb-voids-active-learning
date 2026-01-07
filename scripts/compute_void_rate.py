from pathlib import Path
import cv2, numpy as np, sys

# Dossier de prédictions passé en argument, ex: models/runs/test-pred
pred_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/runs/val-pred")

img_dir = pred_dir / "images"
lbl_dir = pred_dir / "labels"
out_csv = pred_dir.parent / f"{pred_dir.name}_void_rates.csv"

def load_polys(path: Path):
    polys = []
    if not path.exists():
        return polys
    s = path.read_text().strip()
    if not s:
        return polys
    for line in s.splitlines():
        parts = line.split()
        cls = int(parts[0])
        nums = list(map(float, parts[1:]))
        if len(nums) < 6:
            continue
        pts = np.array(list(zip(nums[0::2], nums[1::2])), dtype=np.float32)
        polys.append((cls, pts))
    return polys

rows = ["image,chip_id,chip_area_px,void_area_px,void_rate_percent"]

for img_path in img_dir.glob("*.*"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]
    polys = load_polys(lbl_dir / (img_path.stem + ".txt"))

    # Mapping confirmé : 0 = void, 1 = chip
    chips = [(i, (pts * np.array([w, h])).astype(np.int32))
             for i, (cls, pts) in enumerate(polys) if cls == 1]
    voids = [(pts * np.array([w, h])).astype(np.int32)
             for cls, pts in polys if cls == 0]

    # Masque global des voids
    void_mask = np.zeros((h, w), np.uint8)
    for v in voids:
        cv2.fillPoly(void_mask, [v], 1)

    # Taux de void par chip
    for chip_id, chip_poly in chips:
        chip_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(chip_mask, [chip_poly], 1)
        inter = (void_mask & chip_mask).astype(np.uint8)

        chip_area = int(chip_mask.sum())
        void_area = int(inter.sum())
        void_rate = (void_area / chip_area * 100.0) if chip_area > 0 else 0.0

        rows.append(f"{img_path.name},{chip_id},{chip_area},{void_area},{void_rate:.2f}")

out_csv.write_text("\n".join(rows), encoding="utf-8")
print(f"✓ CSV généré → {out_csv}")
