"""Filtrage par contraintes (chapitre 4) avec relaxation : appliqué en post-traitement de la liste hybride.
Contraintes : prix max, note moyenne min, catégorie. Si trop peu d'items survivent, on relaxe dans l'ordre
(note min -> prix max -> catégorie) et on le signale à l'utilisateur."""
from __future__ import annotations
import numpy as np
import pandas as pd


def apply_constraints(items: pd.DataFrame, cand: np.ndarray, k: int, max_price=None, min_rating=None, category=None):
    """cand : indices d'items triés par score décroissant. Retourne (indices retenus, message de relaxation)."""
    steps = [("note minimale", "min_rating"), ("prix maximal", "max_price"), ("catégorie", "category")]
    active = {"max_price": max_price, "min_rating": min_rating, "category": category}
    relaxed = []

    def ok(idx):
        it = items.loc[idx]
        m = np.ones(len(it), dtype=bool)
        if active["max_price"] is not None:
            m &= (it["price"].fillna(np.inf) <= active["max_price"]).values
        if active["min_rating"] is not None:
            m &= (it["avg_rating"].fillna(0) >= active["min_rating"]).values
        if active["category"] not in (None, "", "Toutes"):
            m &= (it["category"] == active["category"]).values
        return m

    while True:
        mask = ok(cand)
        if mask.sum() >= k or all(active[key] in (None, "", "Toutes") for _, key in steps):
            break
        for label, key in steps:                      # relaxe la première contrainte encore active
            if active[key] not in (None, "", "Toutes"):
                active[key] = None; relaxed.append(label); break
    msg = "" if not relaxed else "Contraintes relaxées pour obtenir assez de résultats : " + ", ".join(relaxed) + "."
    return cand[mask][:k], msg
