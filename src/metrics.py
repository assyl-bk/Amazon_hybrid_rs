"""Métriques d'évaluation (chapitre 6). Fonctions génériques (ensemble pertinent quelconque)
+ métriques au-delà de la précision : couverture, diversité (ILS), nouveauté."""
from __future__ import annotations
import numpy as np


def precision_at_k(rec, relevant, k):
    return len(set(rec[:k]) & set(relevant)) / k

def recall_at_k(rec, relevant, k):
    return len(set(rec[:k]) & set(relevant)) / len(relevant) if len(relevant) else 0.0

def hit_at_k(rec, relevant, k):
    return float(len(set(rec[:k]) & set(relevant)) > 0)

def reciprocal_rank(rec, relevant):
    for r, i in enumerate(rec, 1):
        if i in relevant:
            return 1.0 / r
    return 0.0

def average_precision(rec, relevant, k=None):
    rec = rec[:k] if k else rec
    hits, s = 0, 0.0
    for r, i in enumerate(rec, 1):
        if i in relevant:
            hits += 1; s += hits / r
    return s / len(relevant) if len(relevant) else 0.0

def ndcg_at_k(rec, relevant, k):
    dcg = sum(1.0 / np.log2(r + 1) for r, i in enumerate(rec[:k], 1) if i in relevant)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---- au-delà de la précision --------------------------------------------------------------
def coverage(rec_lists, n_items):
    """Part du catalogue recommandée au moins une fois."""
    return len({i for l in rec_lists for i in l}) / n_items

def novelty(rec_lists, item_pop_frac):
    """Nouveauté moyenne = -log2 p(i), p(i) = proportion d'utilisateurs ayant déjà interagi avec i."""
    p = np.clip(item_pop_frac, 1e-9, 1)
    return float(np.mean([np.mean(-np.log2(p[l])) for l in rec_lists]))

def intra_list_diversity(rec_lists, X):
    """Diversité = 1 - ILS ; ILS = similarité cosinus moyenne entre paires d'items d'une liste.
    X : matrice items x features, lignes L2-normalisées (ici TF-IDF)."""
    vals = []
    for l in rec_lists:
        V = X[l]
        S = (V @ V.T).toarray() if hasattr(V, "toarray") else V @ V.T
        n = len(l)
        if n < 2:
            continue
        ils = (S.sum() - np.trace(S)) / (n * (n - 1))
        vals.append(1 - ils)
    return float(np.mean(vals)) if vals else float("nan")
