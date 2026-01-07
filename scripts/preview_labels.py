from pathlib import Path
import cv2, numpy as np

DATA = Path("data/processed")
OUT  = Path("models/runs/dataset_preview"); OUT.mkdir(parents=True, exist_ok=True)

def draw_polys(img, lbl_path, class_colors):
    H, W = img.shape[:2]
    if not lbl_path.exists(): 
        return img
    txt = lbl_path.read_text().strip()
    if not txt:
        return img
    for line in txt.splitlines():
        parts = line.split()
        cls = int(parts[0]); pts = list(map(float, parts[1:]))
        if len(pts) < 6: 
            continue
        pts = np.array(list(zip(pts[0::2], pts[1::2])), dtype=np.float32)
        pts = (pts * np.array([W, H], dtype=np.float32)).astype(np.int32)
        color = class_colors.get(cls, (255,255,255))
        cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], isClosed=True, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
    return img

def main(split="val", limit=8):
    img_dir = DATA / "images" / split
    lbl_dir = DATA / "labels" / split
    class_colors = {0: (0,255,0), 1: (255,0,0)}  # chip, void (BGR)
    count = 0
    for img_path in sorted(img_dir.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg",".jpeg",".png"}: 
            continue
        img = cv2.imread(str(img_path))
        if img is None: 
            continue
        drawn = draw_polys(img.copy(), lbl_dir / (img_path.stem + ".txt"), class_colors)
        cv2.imwrite(str(OUT / f"{split}_{img_path.name}"), drawn)
        count += 1
        if count >= limit: 
            break
    print(f"✓ Aperçus sauvegardés dans: {OUT.resolve()} (split={split}, n={count})")

if __name__ == "__main__":
    main("val", 8)
