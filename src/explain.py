"""Explications des recommandations (transparence : slide « Avantages » du chapitre 3 / 5)."""
from __future__ import annotations
import numpy as np


def top_shared_terms(content_model, hist_idx, item_idx, n=4):
    """Termes TF-IDF qui contribuent le plus à la similarité entre le profil de l'utilisateur et l'item."""
    X = content_model.X
    prof = np.asarray(X[hist_idx].mean(axis=0)).ravel()
    row = X[item_idx]
    contrib = row.multiply(prof).tocoo()
    if contrib.nnz == 0:
        return []
    order = np.argsort(-contrib.data)[:n]
    vocab = np.array(content_model.vec.get_feature_names_out())
    return [vocab[contrib.col[o]] for o in order if contrib.data[o] > 0]


def explain(item_idx, hist_idx, models: dict, weights: dict, items_df, comp_scores: dict) -> str:
    """comp_scores : {clé modèle: score normalisé de l'item i} déjà pondérables."""
    contrib = {k: weights.get(k, 0) * comp_scores.get(k, 0) for k in weights}
    tot = sum(contrib.values()) or 1.0
    parts = []
    names = {"knn": "collaboratif item-item", "svd": "collaboratif SVD", "content": "contenu", "pop": "popularité"}
    for k, v in sorted(contrib.items(), key=lambda kv: -kv[1]):
        if v / tot >= 0.05:
            parts.append(f"{names.get(k, k)} {100 * v / tot:.0f}%")
    reasons = []
    knn = models["knn"]
    share = {k: v / tot for k, v in contrib.items()}
    if len(hist_idx) and share.get("knn", 0) >= 0.10:
        sims = knn.S[hist_idx][:, item_idx].toarray().ravel()
        if sims.max() > 0:
            j = hist_idx[int(sims.argmax())]
            reasons.append(f"les acheteurs de « {items_df.loc[j, 'title'][:60]} » ont aussi aimé cet article (sim. {sims.max():.2f})")
    if len(hist_idx) and share.get("content", 0) >= 0.10:
        terms = top_shared_terms(models["content"], hist_idx, item_idx)
        if terms:
            reasons.append("mots-clés en commun avec votre profil : " + ", ".join(terms))
    if share.get("pop", 0) >= 0.20:
        reasons.append("article populaire auprès des clients")
    txt = " ; ".join(reasons) if reasons else "recommandation basée sur la popularité"
    return f"{txt.capitalize()}. [Contributions : {', '.join(parts)}]"
