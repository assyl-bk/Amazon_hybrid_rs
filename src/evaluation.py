"""Évaluation hors ligne : classement complet du catalogue (sans échantillonnage de négatifs),
tuning des poids de l'hybride, scénarios cold-start (utilisateurs et items)."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as sp
import config as C
from . import data as D, models as M, metrics as MT


def topk_lists(scores: np.ndarray, H, k: int, pop_tiebreak: np.ndarray):
    """Top-k par utilisateur en excluant l'historique. Départage des ex-aequo par popularité."""
    s = scores.astype(np.float64) + 1e-9 * pop_tiebreak[None, :]
    Hc = H.tocoo()
    s[Hc.row, Hc.col] = -np.inf
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(s, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def _batches(n, bs):
    for s in range(0, n, bs):
        yield s, min(s + bs, n)


def evaluate(model, H: sp.csr_matrix, targets: np.ndarray, k=C.TOPK, pop=None, item_frac=None, X_div=None,
             bs=500, beyond=True) -> dict:
    """H : historiques (n x n_items) ; targets : item cible (un seul item pertinent par utilisateur, leave-one-out).
    NB : avec 1 seul item pertinent, Recall@K = HR@K et Precision@K = HR@K / K."""
    n_items = H.shape[1]
    pop = pop if pop is not None else np.zeros(n_items)
    tb = pop / max(pop.max(), 1)
    res = {"P": [], "R": [], "HR": [], "NDCG": [], "MRR": []}
    lists = []
    for s, e in _batches(H.shape[0], bs):
        L = topk_lists(model.score(H[s:e]), H[s:e], k, tb)
        for row, t in zip(L, targets[s:e]):
            rec = list(row)
            rel = {int(t)}
            res["P"].append(MT.precision_at_k(rec, rel, k)); res["R"].append(MT.recall_at_k(rec, rel, k))
            res["HR"].append(MT.hit_at_k(rec, rel, k)); res["NDCG"].append(MT.ndcg_at_k(rec, rel, k))
            res["MRR"].append(MT.reciprocal_rank(rec, rel))
            lists.append(rec)
    out = {f"Precision@{k}": np.mean(res["P"]), f"Recall@{k}": np.mean(res["R"]), f"HR@{k}": np.mean(res["HR"]),
           f"NDCG@{k}": np.mean(res["NDCG"]), "MRR": np.mean(res["MRR"]), "n": len(targets)}
    if beyond:
        out["Couverture"] = MT.coverage(lists, n_items)
        if item_frac is not None:
            out["Nouveauté"] = MT.novelty(lists, item_frac)
        if X_div is not None:
            out["Diversité"] = MT.intra_list_diversity(lists, X_div)
    return out


# ----------------------------------------------------------------------------- tuning des poids
def weight_grid(keys, step=C.WEIGHT_STEP):
    n = int(round(1 / step))
    for combo in itertools.product(range(n + 1), repeat=len(keys)):
        if sum(combo) == n:
            yield {k: c / n for k, c in zip(keys, combo)}


def tune_weights(models: dict, H, targets, keys=("knn", "svd", "content", "pop"), k=C.TOPK, pop=None,
                 step=C.WEIGHT_STEP, max_users=2000, bs=400, seed=C.SEED):
    """Recherche par grille des poids de l'hybride pondéré (critère : NDCG@K sur la validation)."""
    rng = np.random.default_rng(seed)
    n = H.shape[0]
    sel = np.arange(n) if n <= max_users else np.sort(rng.choice(n, max_users, replace=False))
    H, targets = H[sel], targets[sel]
    grid = list(weight_grid(keys, step))
    tb = (pop / max(pop.max(), 1)) if pop is not None else np.zeros(H.shape[1])
    tot = np.zeros(len(grid)); hr = np.zeros(len(grid))
    for s, e in _batches(H.shape[0], bs):
        comp = {kk: M.row_max_normalize(models[kk].score(H[s:e])) for kk in keys}
        for gi, w in enumerate(grid):
            S = sum(w[kk] * comp[kk] for kk in keys if w[kk] > 0)
            L = topk_lists(S, H[s:e], k, tb)
            for row, t in zip(L, targets[s:e]):
                tot[gi] += MT.ndcg_at_k(list(row), {int(t)}, k)
                hr[gi] += MT.hit_at_k(list(row), {int(t)}, k)
    nd = tot / H.shape[0]
    table = pd.DataFrame(grid); table[f"NDCG@{k}"] = nd; table[f"HR@{k}"] = hr / H.shape[0]
    table = table.sort_values(f"NDCG@{k}", ascending=False).reset_index(drop=True)
    best = {kk: float(table.loc[0, kk]) for kk in keys}
    return best, table


# ----------------------------------------------------------------------------- préparation des historiques
class Split:
    """Regroupe train/val/test + matrices d'historique."""
    def __init__(self, ds: D.Dataset):
        self.ds = ds
        self.train, self.val, self.test = D.temporal_split(ds.inter)
        nU, nI = ds.n_users, ds.n_items
        self.R_train = D.to_csr(self.train, nU, nI)
        self.R_trainval = D.to_csr(pd.concat([self.train, self.val]), nU, nI)
        self.val_target = self.val.set_index("u")["i"].reindex(range(nU)).values
        self.test_target = self.test.set_index("u")["i"].reindex(range(nU)).values
        self.pop = np.asarray(self.R_train.sum(axis=0)).ravel()
        self.item_frac = self.pop / nU
        self.texts = ds.items["text"].tolist()


def evaluate_all(models: dict, hybrids: dict, H, targets, pop, item_frac, X_div, k=C.TOPK) -> pd.DataFrame:
    rows = {}
    for name, m in {**{m.name: m for m in models.values()}, **hybrids}.items():
        rows[name] = evaluate(m, H, targets, k=k, pop=pop, item_frac=item_frac, X_div=X_div)
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- cold-start utilisateurs
def truncate_histories(train: pd.DataFrame, users: np.ndarray, n_keep: int) -> pd.DataFrame:
    """Garde les n_keep PREMIÈRES interactions (chronologiques) des utilisateurs 'froids'."""
    tr = train.sort_values(["u", "timestamp"])
    rank = tr.groupby("u").cumcount()
    is_cold = tr["u"].isin(set(users))
    return tr[~is_cold | (rank < n_keep)]


def cold_user_experiment(sp_: Split, n_keep: int, cold_users: np.ndarray, hybrid_builder, k=C.TOPK) -> pd.DataFrame:
    """Réentraîne tous les modèles avec les historiques des utilisateurs froids tronqués à n_keep,
    puis évalue sur leur interaction de test."""
    ds = sp_.ds
    tr = truncate_histories(sp_.train, cold_users, n_keep)
    R = D.to_csr(tr, ds.n_users, ds.n_items)
    models = M.fit_all(R, sp_.texts)
    H = R[cold_users]
    targets = sp_.test_target[cold_users]
    pop = np.asarray(R.sum(axis=0)).ravel()
    hybrids = hybrid_builder(models)
    return evaluate_all(models, hybrids, H, targets, pop, pop / ds.n_users, models["content"].X, k)


# ----------------------------------------------------------------------------- cold-start items
def cold_item_experiment(sp_: Split, frac: float, hybrid_builder, k=C.TOPK, seed=C.SEED):
    """Simule de NOUVEAUX items : `frac` des items sont retirés de l'apprentissage ; on évalue les utilisateurs
    dont l'interaction de test porte sur un de ces items (aucun CF possible, seul le contenu peut les trouver)."""
    ds = sp_.ds
    rng = np.random.default_rng(seed)
    cold_items = set(rng.choice(ds.n_items, int(frac * ds.n_items), replace=False))
    tr = sp_.train[~sp_.train["i"].isin(cold_items)]
    trv = pd.concat([sp_.train, sp_.val]); trv = trv[~trv["i"].isin(cold_items)]
    R = D.to_csr(tr, ds.n_users, ds.n_items)
    H_all = D.to_csr(trv, ds.n_users, ds.n_items)
    models = M.fit_all(R, sp_.texts)
    users = np.where(np.isin(sp_.test_target, list(cold_items)))[0]
    pop = np.asarray(R.sum(axis=0)).ravel()
    return models, users, H_all, pop, cold_items
