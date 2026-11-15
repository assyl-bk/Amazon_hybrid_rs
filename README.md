# Projet RS2026 — Recommandeur hybride sur Amazon Reviews 2023

Hybride **pondéré + switching** (CF item-item, CF SVD, contenu TF-IDF, popularité) avec filtres par contraintes,
évaluation complète (précision, couverture, diversité, nouveauté, cold-start) et interface Gradio avec explications.

---

## 0. Prérequis
* Python **3.10 ou plus récent** (vérifier : `python --version`)
* ~2 Go d'espace disque libre et une connexion internet pour le téléchargement des données

## 1. Installation
Ouvre un terminal **dans le dossier `amazon_hybrid_rs`** (celui qui contient ce README).
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Mac / Linux
pip install -r requirements.txt
python tests/test_metrics.py      # doit afficher : OK
```

## 2. Télécharger les données (gratuit, sans compte)
Dataset : **Amazon Reviews 2023** (McAuley Lab) — https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
⚠️ Le dépôt complet pèse **750 Go** : ne le télécharge JAMAIS en entier. On ne prend que 2 fichiers par catégorie.
Catégories par défaut (modifiables dans `config.py`) : `Video_Games`, `Software`, `Musical_Instruments`.

### 2.A Méthode automatique (recommandée)
```bash
python scripts/download_data.py
```
Le script télécharge pour chaque catégorie : le CSV d'interactions 5-core (`rating_only`, avec repli sur `last_out`)
et les métadonnées produits (`.jsonl` depuis Hugging Face). Les fichiers déjà présents sont ignorés. Autres options :
```bash
python scripts/download_data.py --cats Video_Games Software Musical_Instruments
python scripts/download_data.py --mode raw --cats Digital_Music Software    # avis bruts, plus lourd (k-core fait par notre code)
```
Si une ligne affiche « ✗ échec », passe à la méthode manuelle (2.B) pour le fichier concerné.

### 2.B Méthode manuelle (navigateur)
> L'ancien miroir `datarepo.eng.ucsd.edu/...jsonl.gz` renvoie **404**. Les fichiers de métadonnées sont sur Hugging Face, en **`.jsonl` non compressé**.

1. **Interactions (CSV 5-core)**
   * Page du dataset → onglet **Files and versions** → dossier `benchmark` → `5core` → `rating_only`.
   * Clique sur `Video_Games.csv` puis sur l'icône de téléchargement (↓). Répète pour `Software.csv` et `Musical_Instruments.csv`.
   * Si `rating_only` n'existe pas : dossier `last_out` → `<Categorie>.train.csv`, `.valid.csv`, `.test.csv` (3 fichiers par catégorie ; le code les recolle).
2. **Métadonnées produits** — onglet **Files and versions** → dossier `raw` → `meta_categories` → clique sur le fichier puis sur ↓ :
   * `meta_Video_Games.jsonl`
   * `meta_Software.jsonl` (≈ 256 Mo)
   * `meta_Musical_Instruments.jsonl`

   Liens directs (testés par recherche, pas téléchargés par moi) :
   * https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Video_Games.jsonl
   * https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Software.jsonl
   * https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Musical_Instruments.jsonl

   Autre catégorie : remplace le nom dans l'URL. Le code accepte `.jsonl` **ou** `.jsonl.gz` : **ne décompresse/renomme rien**.
3. **Rangement** : place tous les fichiers dans `data/raw/` (aucun sous-dossier).

### 2.C À quoi servent ces fichiers ?
| Fichier | Contenu | Rôle dans le projet |
|---|---|---|
| `<Categorie>.csv` (interactions) | `user_id, parent_asin, rating, timestamp` : qui a noté quel produit, quand | Matrice utilisateur–item : **filtrage collaboratif** (item-kNN, SVD), popularité, split temporel, évaluation |
| `meta_<Categorie>.jsonl` (métadonnées) | titre, catégories, caractéristiques, description, prix, note moyenne, marque | **Approche contenu (TF-IDF)**, filtres prix/note de l'interface, noms affichés, explications, mesure de diversité |

Sans les métadonnées le projet ne tourne pas : le code ne garde que les produits qui en ont une. Sans les interactions, il n'y a rien à recommander.

### 2.D Vérification
`data/raw/` doit contenir (les noms doivent correspondre EXACTEMENT aux catégories de `config.py`) :
```
Video_Games.csv              meta_Video_Games.jsonl
Software.csv                 meta_Software.jsonl
Musical_Instruments.csv      meta_Musical_Instruments.jsonl
```
(ou `.jsonl.gz` pour les métadonnées ; ou `<Categorie>.train/valid/test.csv` à la place du `.csv` unique.)

## 3. Lancer le projet
### 3.A Avec le notebook (livrable principal)
```bash
jupyter notebook notebooks/projet_rs_hybride.ipynb
```
* Vérifie que `USE_SYNTHETIC = False` dans la première cellule de code.
* Menu **Kernel → Restart & Run All**. Le notebook enchaîne : statistiques des données → split → modèles de base → réglage des poids
  de l'hybride → évaluation → cold-start utilisateurs et items → aperçu des explications.
* Il écrit les résultats (`artifacts/results/*.csv`) et `artifacts/bundle.pkl` (nécessaire à l'interface).

### 3.B Sans notebook (terminal)
```bash
python scripts/run_experiments.py
```

### 3.C Interface de démonstration
```bash
python app.py
```
Ouvre http://127.0.0.1:7860 : utilisateur existant (avec simulation de cold-start), nouvel utilisateur par mots-clés,
filtres prix / note / catégorie (relaxation automatique) et explication de chaque recommandation.
Arrêt : `Ctrl+C` dans le terminal.

## 4. Si ça coince
| Problème | Solution |
|---|---|
| `Aucune donnée pour '<Categorie>' / Métadonnées manquantes` | Fichier absent de `data/raw/` ou nom différent de la catégorie dans `config.py` |
| Trop peu d'utilisateurs après filtrage | Baisse `MIN_USER_INTER` / `N_USERS_PER_CAT` dans `config.py`, ou change de catégories |
| Exécution trop lente / mémoire saturée | Mets `N_USERS_PER_CAT = 1000` (ou moins) dans `config.py` |
| `app.py` : `bundle.pkl` introuvable | Exécute d'abord le notebook (3.A) ou `run_experiments.py` (3.B) |
| Échec / 404 au téléchargement automatique | Utilise la méthode manuelle 2.B (l'ancien miroir UCSD `.jsonl.gz` est mort ; les fichiers sont sur Hugging Face en `.jsonl`) |
| Vérifier que le code tourne sans données | Mets `USE_SYNTHETIC = True` dans le notebook (données FACTICES, résultats sans valeur) |

## 5. Structure du projet
```
config.py                 paramètres (catégories, seuils, K, poids…)
src/data.py               chargement, positifs (note>=4), k-core, échantillon équilibré, split temporel
src/models.py             Popularité, ItemKNN, SVD, Contenu TF-IDF, HybridWeighted, HybridSwitch
src/metrics.py            P@K, R@K, HR@K, NDCG@K, MRR, AP, couverture, diversité (ILS), nouveauté
src/evaluation.py         évaluation, tuning des poids, cold-start utilisateurs / items
src/knowledge.py          filtres par contraintes avec relaxation (chap. 4)
src/explain.py            explications des recommandations
src/pipeline.py           orchestration + sauvegarde des artefacts
scripts/                  download_data.py, run_experiments.py, make_synthetic.py (tests), build_notebook.py
tests/test_metrics.py     vérifie les métriques contre les exemples du cours
notebooks/                notebook du projet (suit l'énoncé point par point)
app.py                    interface Gradio
rapport/PLAN_RAPPORT.md   plan du rapport (10 pages max)
ia_historique/            à remplir : export COMPLET des conversations avec l'IA + journal
```

## 6. Livrables demandés par l'énoncé
1. Notebook + code source commenté → `notebooks/` + `src/`
2. Rapport 10 pages max → à rédiger à partir de `rapport/PLAN_RAPPORT.md`
3. Historique complet des conversations avec l'IA → `ia_historique/` (voir `LISEZMOI.md` dans ce dossier)

