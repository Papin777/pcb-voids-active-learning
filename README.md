# 🔍 PCB Void Detection & Active Learning (YOLO + SAM)

Ce projet propose une **application de détection et de quantification de voids (trous) sur des images X-ray de PCB**, basée sur une approche **YOLO + SAM**, intégrant un **processus d’Active Learning** et un **déploiement cloud sur Azure**.

---

## 🎯 Objectifs du projet

- Détecter automatiquement :
  - les **chips** (composants électroniques)
  - les **voids** (trous / défauts)
- Calculer des **métriques de surface** :
  - aire des composants
  - aire des voids
  - **taux de void par composant**
- Permettre à l’utilisateur de :
  - corriger les prédictions
  - ajouter de nouvelles annotations
  - **réentraîner le modèle (Active Learning)**
- Déployer l’application sous forme de **service cloud accessible via une interface web**

---

## 🧠 Architecture IA

### Modèles utilisés
- **YOLO11-seg (Ultralytics)**  
  → Détection + segmentation initiale des chips et voids
- **Segment Anything Model (SAM)**  
  → Raffinement des masques et annotation manuelle assistée

### Pipeline général
1. Prédiction automatique YOLO (chips / voids)
2. Raffinement ou correction avec SAM
3. Calcul des surfaces et taux de void
4. Sauvegarde des corrections
5. Réentraînement du modèle (Active Learning)

---

## 🖥️ Application Streamlit

L’application permet :
- 📤 Upload d’images X-ray
- 👁️ Visualisation des masques YOLO / SAM
- ✏️ Correction manuelle des prédictions
- 📊 Analyse des taux de void
- 📁 Export des résultats en CSV
- 🔁 Réentraînement du modèle depuis l’interface

Fichier principal :
```bash
streamlit_sam_active_learning.py

pcb-voids-active-learning/
│
├── app/                    # Logique UI Streamlit (pages)
├── scripts/                # Scripts d’analyse (void rate, CSV, etc.)
├── data/                   # Dossiers de données (hors Git)
├── checkpoints/            # Modèles SAM (hors Git)
├── models/                 # Modèles YOLO entraînés (hors Git)
│
├── streamlit_sam_active_learning.py
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
├── .gitignore
├── .dockerignore
└── README.md
