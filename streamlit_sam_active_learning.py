import io
from pathlib import Path
import subprocess  # pour lancer YOLO depuis Streamlit

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

from segment_anything import sam_model_registry, SamPredictor

# ============================================================
# Config modèles
# ============================================================

# YOLO : ton meilleur modèle entraîné
YOLO_MODEL_PATH = Path("models/runs/voids_active_learning_from_app/weights/best.pt")

# SAM : checkpoint à placer dans checkpoints/sam_vit_b.pth
SAM_CHECKPOINT = Path("checkpoints/sam_vit_b.pth")
SAM_MODEL_TYPE = "vit_b"   # ou "vit_h" / "vit_l" si tu as d'autres poids

# Mapping réel du modèle YOLO : 0 = chip, 1 = void
CLASS_NAMES = {0: "chip", 1: "void"}

# Dossiers pour l'active learning
CORR_LABELS_DIR = Path("data/corrections/labels")
ACTIVE_DATA_DIR = Path("data/active_learning")


def prepare_active_learning_yaml() -> Path:
    """
    Crée (ou écrase) data/active_learning/data.yaml pour réentraîner YOLO
    à partir des corrections, avec validation sur le dataset original.
    """
    ACTIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    yaml_text = """train: ../corrections/images
val: ../processed/images/val
test: ../processed/images/test

names:
  0: chip
  1: void
"""
    yaml_path = ACTIVE_DATA_DIR / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


@st.cache_resource
def load_yolo_model():
    return YOLO(str(YOLO_MODEL_PATH))


@st.cache_resource
def load_sam_predictor():
    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint SAM introuvable: {SAM_CHECKPOINT}")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    predictor = SamPredictor(sam)
    return predictor


# ============================================================
# Utils
# ============================================================

def yolo_infer(image_rgb: np.ndarray, conf: float, iou: float):
    model = load_yolo_model()
    results = model(image_rgb, imgsz=512, conf=conf, iou=iou, verbose=False)
    return results[0]  # un seul batch


def extract_instances(result):
    """
    Transforme le résultat YOLO en liste de dictionnaires:
    [{"id": int, "cls": int, "name": str, "box_xyxy": np.ndarray[4], "poly": np.ndarray[N,2]}, ...]
    """
    instances = []
    if result.masks is None or result.boxes is None:
        return instances

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
    classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
    polys = result.masks.xy  # liste de N tableaux (Mi, 2)

    for idx, (cls_id, box, poly) in enumerate(zip(classes, boxes_xyxy, polys)):
        instances.append(
            {
                "id": idx,
                "cls": int(cls_id),
                "name": CLASS_NAMES.get(int(cls_id), f"class_{cls_id}"),
                "box_xyxy": np.array(box, dtype=np.float32),
                "poly": np.array(poly, dtype=np.float32),
            }
        )
    return instances


def draw_instances_overlay(img_rgb: np.ndarray, instances):
    """
    Dessine tous les masques sur une copie de l'image.
    - chip (0) en bleu
    - void (1) en vert
    """
    overlay = img_rgb.copy()
    for inst in instances:
        poly = inst["poly"].astype(np.int32)
        cls_id = inst["cls"]
        if cls_id == 1:      # void
            color = (0, 255, 0)   # vert
        else:                # chip
            color = (0, 0, 255)   # bleu
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(overlay, [poly], True, (0, 0, 0), 1)
    vis = cv2.addWeighted(overlay, 0.45, img_rgb, 0.55, 0)
    return vis


def compute_void_rates(img_rgb: np.ndarray, instances):
    """
    Calcule le void rate par chip.
    Mapping :
      - YOLO classe 0 = chip
      - YOLO classe 1 = void
    """
    h, w = img_rgb.shape[:2]
    chips = [(inst["id"], inst["poly"]) for inst in instances if inst["cls"] == 0]
    voids = [inst["poly"] for inst in instances if inst["cls"] == 1]

    if not chips:
        return pd.DataFrame(columns=["chip_id", "chip_area_px", "void_area_px", "void_rate_percent"])

    void_mask = np.zeros((h, w), np.uint8)
    for poly in voids:
        cv2.fillPoly(void_mask, [poly.astype(np.int32)], 1)

    rows = []
    for chip_id, chip_poly in chips:
        chip_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(chip_mask, [chip_poly.astype(np.int32)], 1)
        inter = (void_mask & chip_mask).astype(np.uint8)

        chip_area = int(chip_mask.sum())
        void_area = int(inter.sum())
        void_rate = (void_area / chip_area * 100.0) if chip_area > 0 else 0.0

        rows.append(
            {
                "chip_id": chip_id,
                "chip_area_px": chip_area,
                "void_area_px": void_area,
                "void_rate_percent": round(void_rate, 2),
            }
        )

    return pd.DataFrame(rows)


def sam_refine_mask(img_rgb: np.ndarray, box_xyxy: np.ndarray):
    """
    Utilise SAM pour générer un nouveau masque à partir de la bounding box YOLO.
    Retourne un masque binaire (H,W) uint8 {0,1}.
    """
    predictor = load_sam_predictor()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    predictor.set_image(img_bgr)

    box = box_xyxy[None, :]  # (1,4)
    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(np.uint8)  # (H,W) 0/1
    return best_mask


def mask_to_polygon(mask: np.ndarray):
    """
    Convertit un masque binaire (H,W) en un polygon approx (N,2) en coords (x,y).
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    largest = largest.squeeze(1)  # (N,2)
    return largest.astype(np.float32)


def save_corrected_annotation(image_name: str, instances, img_shape, save_dir: Path):
    """
    Sauvegarde les instances (polygones) au format YOLO segmentation dans save_dir/labels
    et attend que l'image soit copiée à côté (pour réentraînement).
    img_shape: (H,W,3)
    """
    h, w = img_shape[:2]
    save_lbl_dir = save_dir / "labels"
    save_lbl_dir.mkdir(parents=True, exist_ok=True)

    label_path = save_lbl_dir / (Path(image_name).stem + ".txt")

    lines = []
    for inst in instances:
        cls_id = inst["cls"]
        poly = inst["poly"]  # (N,2) en pixels
        if poly is None or len(poly) < 3:
            continue
        # Normaliser en [0,1]
        xs = poly[:, 0] / w
        ys = poly[:, 1] / h
        coords = []
        for x, y in zip(xs, ys):
            coords.extend([x, y])
        line = f"{cls_id} " + " ".join(f"{c:.6f}" for c in coords)
        lines.append(line)

    label_path.write_text("\n".join(lines), encoding="utf-8")
    return label_path


# ============================================================
# UI Streamlit
# ============================================================

st.set_page_config(page_title="Void Detection + SAM Active Learning", layout="wide")
st.title("🔍 Void Detection & Active Learning (YOLO + SAM)")

# Navigation "pages"
page = st.sidebar.selectbox(
    "Navigation",
    ["Analyse & CSV", "Correction + SAM", "Active Learning"],
    index=0,
)

st.markdown(
    """
    **Pipeline :**
    1. YOLO11-seg prédit les **chips** (bleu) et les **voids** (vert)  
    2. Tu peux corriger une instance avec **SAM**  
    3. Tu peux **sauvegarder les annotations corrigées** pour un futur réentraînement  
    4. Tu peux lancer un **réentraînement YOLO** à partir des corrections (active learning)
    """
)

if not YOLO_MODEL_PATH.exists():
    st.error(f"Modèle YOLO introuvable : `{YOLO_MODEL_PATH}`")
    st.stop()

if not SAM_CHECKPOINT.exists():
    st.warning(f"Checkpoint SAM introuvable : `{SAM_CHECKPOINT}`. Télécharge-le avant d'utiliser la correction SAM.")

conf = st.sidebar.slider("Seuil de confiance YOLO", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("Seuil IoU NMS YOLO", 0.1, 0.9, 0.6, 0.05)

# Uploader seulement utile pour Analyse & Correction
uploaded = None
img_rgb = None
instances = None
df_void = None
vis = None

if page in ["Analyse & CSV", "Correction + SAM"]:
    uploaded = st.file_uploader("Charge une image X-ray", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        # lecture image
        bytes_data = uploaded.read()
        img_array = np.frombuffer(bytes_data, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Inférence YOLO commune aux deux pages
        with st.spinner("Inférence YOLO..."):
            yolo_res = yolo_infer(img_rgb, conf=conf, iou=iou)
            instances = extract_instances(yolo_res)

        if instances:
            vis = draw_instances_overlay(img_rgb, instances)
            df_void = compute_void_rates(img_rgb, instances)
        else:
            st.warning("Aucune instance détectée (ni chip ni void).")

# ===========================
# PAGE 1 : Analyse & CSV
# ===========================
if page == "Analyse & CSV":
    st.header("📊 Analyse des voids & export CSV")

    if uploaded is None:
        st.info("Commence par charger une image X-ray pour lancer l'analyse.")
    elif not instances:
        st.warning("Impossible de calculer le taux de void : aucune instance détectée.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Image originale")
            st.image(img_rgb, use_container_width=True)
        with col2:
            st.subheader("Prédictions YOLO (chips / voids)")
            st.image(vis, caption="Bleu = chip, Vert = void", use_container_width=True)

        st.subheader("Taux de void par chip")
        st.dataframe(df_void, use_container_width=True)

        # Résumé + export CSV
        mean_void = df_void["void_rate_percent"].mean()
        max_void = df_void["void_rate_percent"].max()

        st.markdown("**Résumé :**")
        st.write(f"- Taux de void moyen : **{mean_void:.2f}%**")
        st.write(f"- Taux de void max : **{max_void:.2f}%**")

        csv_bytes = df_void.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger les résultats (CSV)",
            data=csv_bytes,
            file_name="void_rates.csv",
            mime="text/csv",
        )

# ===========================
# PAGE 2 : Correction + SAM
# ===========================
elif page == "Correction + SAM":
    st.header("🛠 Correction d'une instance avec SAM")

    if uploaded is None:
        st.info("Charge une image X-ray pour pouvoir corriger une instance.")
    elif not instances:
        st.warning("Aucune instance détectée à corriger.")
    else:
        # Liste des instances au choix
        options = [
            f"ID {inst['id']} - {inst['name']} - box=({int(inst['box_xyxy'][0])},{int(inst['box_xyxy'][1])},{int(inst['box_xyxy'][2])},{int(inst['box_xyxy'][3])})"
            for inst in instances
        ]
        selected = st.selectbox("Choisir une instance à corriger", options)
        selected_id = int(selected.split()[1])

        current_inst = next(inst for inst in instances if inst["id"] == selected_id)

        st.write(f"Classe actuelle : **{current_inst['name']}** (id={current_inst['cls']})")

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Masque YOLO actuel")
            img_yolo_only = img_rgb.copy()
            cv2.fillPoly(img_yolo_only, [current_inst["poly"].astype(np.int32)], (0, 0, 255))
            st.image(img_yolo_only, use_container_width=True)

        with c2:
            if SAM_CHECKPOINT.exists():
                st.caption("Masque SAM (à partir de la bbox YOLO)")
                with st.spinner("SAM en cours..."):
                    sam_mask = sam_refine_mask(img_rgb, current_inst["box_xyxy"])
                img_sam_only = img_rgb.copy()
                poly_sam = mask_to_polygon(sam_mask)
                if poly_sam is not None:
                    cv2.fillPoly(img_sam_only, [poly_sam.astype(np.int32)], (0, 255, 0))
                st.image(img_sam_only, use_container_width=True)
            else:
                sam_mask = None
                poly_sam = None
                st.info("SAM n'est pas disponible (checkpoint manquant).")

        # Choix de la version à garder
        if sam_mask is not None and poly_sam is not None:
            choice = st.radio(
                "Quel masque veux-tu garder pour cette instance ?",
                ["Masque YOLO (original)", "Masque SAM (raffiné)"],
                index=0,
            )

            if choice == "Masque SAM (raffiné)":
                # on remplace le poly de l'instance par celui de SAM
                current_inst["poly"] = poly_sam

        st.markdown("---")
        st.header("💾 Sauvegarder cette image + annotations corrigées pour réentraînement")

        save_dir = Path("data/corrections")
        if st.button("Sauvegarder les annotations corrigées"):
            label_path = save_corrected_annotation(
                image_name=uploaded.name,
                instances=instances,
                img_shape=img_rgb.shape,
                save_dir=save_dir,
            )

            # Sauvegarde de l'image dans data/corrections/images
            save_img_dir = save_dir / "images"
            save_img_dir.mkdir(parents=True, exist_ok=True)
            out_img_path = save_img_dir / uploaded.name
            cv2.imwrite(str(out_img_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            st.success(f"Annotations sauvegardées dans : {label_path}")
            st.info("Tu pourras inclure data/corrections dans un futur réentraînement YOLO (active learning).")

# ===========================
# PAGE 3 : Active Learning
# ===========================
elif page == "Active Learning":
    st.header("🔁 Active Learning : Retrain YOLO with corrections")

    num_corr = len(list(CORR_LABELS_DIR.glob("*.txt")))
    st.write(f"Nombre de fichiers de correction disponibles : **{num_corr}**")

    if num_corr == 0:
        st.info(
            "Aucune correction trouvée dans data/corrections/labels. "
            "Corrige d'abord au moins une image avec SAM puis sauvegarde."
        )
    else:
        st.write("Le réentraînement utilisera :")
        st.markdown(
            """
            - **train** : `data/corrections/images`  
            - **val**   : `data/processed/images/val`  
            - **test**  : `data/processed/images/test`  
            - **model de départ** : `models/runs/voids_active_learning_from_app/weights/best.pt`
            """
        )

        if st.button("🔁 Lancer le réentraînement YOLO avec les corrections"):
            yaml_path = prepare_active_learning_yaml()
            st.info(f"Fichier de config actif : `{yaml_path}`")

            cmd = [
                "yolo",
                "train",
                "task=segment",
                f"model={str(YOLO_MODEL_PATH)}",
                f"data={str(yaml_path)}",
                "epochs=10",
                "imgsz=512",
                "batch=2",
                "device=cpu",
                "workers=0",
                "amp=False",
                "project=models/runs",
                "name=voids_active_learning_from_app",
            ]

            with st.spinner("Réentraînement YOLO en cours..."):
                result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                st.error("❌ Réentraînement YOLO échoué.")
                st.code(result.stderr)
            else:
                st.success("✅ Réentraînement terminé.")
                st.code("models/runs/voids_active_learning_from_app/weights/best.pt")
