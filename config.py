"""Configuration centrale du projet (modifiable ici ou via variables d'environnement)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

# --- Données -----------------------------------------------------------------
# 3 catégories Amazon Reviews 2023 -> catalogue "équilibré" par catégorie.
CATEGORIES = os.environ.get("RS_CATEGORIES", "Video_Games,Software,Musical_Instruments").split(",")
N_USERS_PER_CAT = int(os.environ.get("RS_USERS_PER_CAT", 2000))  # utilisateurs échantillonnés par catégorie
POS_THRESHOLD = 4        # note >= 4  => interaction "appréciée" (feedback implicite positif)
MIN_USER_INTER = 5       # k-core utilisateurs
MIN_ITEM_INTER = 3       # k-core items (après échantillonnage)
SEED = 42

# --- Modèles -----------------------------------------------------------------
TOPK = 10                # K pour les métriques @K
K_NEIGHBORS = 50         # voisins conservés par item (item-kNN)
SVD_DIM = 64             # dimension latente (SVD)
TFIDF_MAX_FEATURES = 30000
COLD_T = 5               # seuil de commutation : < COLD_T interactions => utilisateur "froid"
WEIGHT_STEP = 0.1        # pas de la grille de pondération
