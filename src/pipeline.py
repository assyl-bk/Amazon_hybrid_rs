"""Orchestration : construction du jeu de données, entraînement, tuning, expériences, sauvegarde des artefacts."""
from __future__ import annotations
import json, pickle
import numpy as np
import pandas as pd
import config as C
from . import data as D, models as M, evaluation as E

KEYS = ("knn", "svd", "content", "pop")


def make_hybrids(models, w_warm, w_cold):
    single = M.HybridWeighted(models, w_warm, item_fallback=False); single.name = "Hybride pondéré"
    sw = M.HybridSwitch(models, w_cold, w_warm); sw.name = "Hybride switching"
    return {single.name: single, sw.name: sw}


def run_experiments(ds: D.Dataset, verbose=True, cold_ns=(1, 2, 3), cold_item_frac=0.05, save=True):
    rng = np.random.default_rng(C.SEED)
    sp_ = E.Split(ds)
    k = C.TOPK
    res = {"split": sp_}

    # 1) Modèles de base entraînés sur train ------------------------------------------------------
    models = M.fit_all(sp_.R_train, sp_.texts)
    X_div = models["content"].X

    # 2) Tuning des poids (grille) sur la VALIDATION : historique = train, cible = avant-dernier item ---
    w_warm, tab_warm = E.tune_weights(models, sp_.R_train, sp_.val_target, KEYS, pop=sp_.pop)
    if verbose: print("Poids optimaux (utilisateurs 'chauds') :", w_warm)

    # 3) Poids 'froids' : 30 % des utilisateurs, historique tronqué à 2 interactions ------------------
    perm = rng.permutation(ds.n_users)
    cold_users = perm[: int(0.3 * ds.n_users)]
    tune_u, test_u = np.sort(cold_users[: len(cold_users) // 2]), np.sort(cold_users[len(cold_users) // 2:])
    tr2 = E.truncate_histories(sp_.train, cold_users, 2)
    R2 = D.to_csr(tr2, ds.n_users, ds.n_items)
    models2 = M.fit_all(R2, sp_.texts)
    w_cold, tab_cold = E.tune_weights(models2, R2[tune_u], sp_.test_target[tune_u], KEYS,
                                      pop=np.asarray(R2.sum(axis=0)).ravel())
    if verbose: print("Poids optimaux (utilisateurs 'froids') :", w_cold)
    res.update(w_warm=w_warm, w_cold=w_cold, tab_warm=tab_warm, tab_cold=tab_cold, cold_test_users=test_u)

    # 4) Évaluation TEST (utilisateurs standards) : historique = train + validation -------------------
    hyb = make_hybrids(models, w_warm, w_cold)
    res["main"] = E.evaluate_all(models, hyb, sp_.R_trainval, sp_.test_target, sp_.pop, sp_.item_frac, X_div, k)
    if verbose: print(res["main"].round(4))

    # 5) Cold-start utilisateurs : réentraînement avec historiques tronqués (1, 2, 3 interactions) ;
    #    évaluation sur les utilisateurs de TEST (disjoints de ceux ayant servi au tuning des poids froids)
    cold_tables = {}
    for n in cold_ns:
        tr_n = E.truncate_histories(sp_.train, cold_users, n)
        R_n = D.to_csr(tr_n, ds.n_users, ds.n_items)
        m_n = M.fit_all(R_n, sp_.texts)
        pop_n = np.asarray(R_n.sum(axis=0)).ravel()
        cold_tables[n] = E.evaluate_all(m_n, make_hybrids(m_n, w_warm, w_cold), R_n[test_u], sp_.test_target[test_u],
                                        pop_n, pop_n / ds.n_users, m_n["content"].X, k)
    res["cold_users"] = cold_tables

    # 6) Cold-start items ---------------------------------------------------------------------------
    cm, users_t, H_all, pop_c, cold_items = E.cold_item_experiment(sp_, cold_item_frac, None, k)
    hyb_ci = make_hybrids(cm, w_warm, w_cold)
    if len(users_t):
        res["cold_items"] = E.evaluate_all(cm, hyb_ci, H_all[users_t], sp_.test_target[users_t], pop_c, pop_c / ds.n_users,
                                           cm["content"].X, k)
    else:
        res["cold_items"] = pd.DataFrame()
    res["cold_items_n_users"] = len(users_t)

    res["models"] = models
    res["hybrids"] = hyb
    if save:
        save_artifacts(ds, sp_, models, w_warm, w_cold, res)
    return res


def save_artifacts(ds, sp_, models, w_warm, w_cold, res):
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(C.ARTIFACTS / "bundle.pkl", "wb") as f:
        pickle.dump({"ds_items": ds.items, "n_users": ds.n_users, "n_items": ds.n_items,
                     "inter": ds.inter[["user_id", "item_id", "rating", "timestamp", "u", "i", "category"]],
                     "models": models, "w_warm": w_warm, "w_cold": w_cold,
                     "R_trainval": sp_.R_trainval, "test_target": sp_.test_target, "pop": sp_.pop}, f)
    out = C.ARTIFACTS / "results"; out.mkdir(exist_ok=True)
    res["main"].to_csv(out / "resultats_principaux.csv")
    for n, t in res["cold_users"].items():
        t.to_csv(out / f"cold_start_utilisateurs_{n}.csv")
    if len(res["cold_items"]):
        res["cold_items"].to_csv(out / "cold_start_items.csv")
    res["tab_warm"].head(15).to_csv(out / "grille_poids_chauds.csv", index=False)
    res["tab_cold"].head(15).to_csv(out / "grille_poids_froids.csv", index=False)
    with open(out / "poids.json", "w") as f:
        json.dump({"warm": w_warm, "cold": w_cold}, f, indent=2)


def load_bundle():
    with open(C.ARTIFACTS / "bundle.pkl", "rb") as f:
        return pickle.load(f)
