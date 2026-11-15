"""Interface de démonstration (Gradio) : recommandations hybrides + explications + contraintes.

Prérequis : avoir exécuté le notebook (ou `python scripts/run_experiments.py`) pour produire artifacts/bundle.pkl.
Lancement :  python app.py   puis ouvrir http://127.0.0.1:7860
"""
import numpy as np, pandas as pd, scipy.sparse as sp
import gradio as gr
import config as C
from src import pipeline as P, models as M, explain as X, knowledge as K

B = P.load_bundle()
items, models, inter = B["ds_items"], B["models"], B["inter"]
w_warm, w_cold = B["w_warm"], B["w_cold"]
n_items = B["n_items"]
hybrid = M.HybridSwitch(models, w_cold, w_warm)
by_user = {u: g["i"].tolist() for u, g in inter.sort_values(["u", "timestamp"]).groupby("u")}   # historique chronologique
test_target = B["test_target"]
CATS = ["Toutes"] + sorted(items["category"].dropna().unique().tolist())

rng = np.random.default_rng(C.SEED)
demo_users = sorted(rng.choice(list(by_user), min(150, len(by_user)), replace=False).tolist())
user_choices = [(f"Utilisateur {u}  ({len(by_user[u])} interactions)", int(u)) for u in demo_users]


def _row(idx, rank, score, why):
    it = items.loc[idx]
    price = "" if pd.isna(it["price"]) else f"{it['price']:.2f} $"
    return {"#": rank, "Produit": it["title"][:90], "Catégorie": it["category"], "Prix": price,
            "Note moy.": it["avg_rating"], "Score": round(float(score), 3), "Pourquoi ce produit ?": why}


def _hist_df(hist):
    rows = [{"Produit": items.loc[i, "title"][:90], "Catégorie": items.loc[i, "category"],
             "Prix": "" if pd.isna(items.loc[i, "price"]) else f"{items.loc[i, 'price']:.2f} $"} for i in hist[-15:]]
    return pd.DataFrame(rows)


def recommend_existing(user, k, simulate, n_keep, max_price, min_rating, category):
    u = int(user)
    full = by_user[u]
    hist_all = full[:-1]                       # tout sauf l'interaction de test (cachée)
    hist = hist_all[: int(n_keep)] if simulate else hist_all
    H = sp.csr_matrix((np.ones(len(hist)), (np.zeros(len(hist), int), hist)), shape=(1, n_items))
    s = hybrid.score(H)[0].copy()
    s[hist] = -np.inf
    cand = np.argsort(-s)[: int(k) * 25]
    sel, msg = K.apply_constraints(items, cand, int(k), max_price or None, min_rating or None, category)
    w = hybrid.weights_for(len(hist))
    comp = M.HybridWeighted(models, w).components(H)
    rows = [_row(i, r + 1, s[i], X.explain(i, np.array(hist), models, w, items, {kk: v[0, i] for kk, v in comp.items()}))
            for r, i in enumerate(sel)]
    mode = "froid (historique court → plus de contenu/popularité)" if hybrid.is_cold(H)[0] else "chaud (collaboratif + contenu)"
    target = int(test_target[u])
    hit = "✅ présent dans les recommandations !" if target in set(sel) else "absent de la liste (ce n'est pas forcément une erreur : cf. limites de l'évaluation hors ligne)"
    info = (f"**Mode de l'hybride :** {mode} — poids {w}\n\n"
            f"**Article réellement acheté ensuite (cible cachée) :** {items.loc[target, 'title'][:100]} → {hit}\n\n{msg}")
    return _hist_df(hist), pd.DataFrame(rows), info


def recommend_new_user(query, k, max_price, min_rating, category):
    if not query or not query.strip():
        return pd.DataFrame(), "Saisis quelques mots-clés (ex. « guitar strings », « racing game »)."
    c = M.row_max_normalize(models["content"].score_query(query))[0]
    pop = models["pop"].s / models["pop"].s.max()
    s = 0.8 * c + 0.2 * pop
    cand = np.argsort(-s)[: int(k) * 25]
    sel, msg = K.apply_constraints(items, cand, int(k), max_price or None, min_rating or None, category)
    q = models["content"].vec.transform([query])
    vocab = np.array(models["content"].vec.get_feature_names_out())
    rows = []
    for r, i in enumerate(sel):
        contrib = models["content"].X[i].multiply(q).tocoo()
        terms = [vocab[contrib.col[o]] for o in np.argsort(-contrib.data)[:4]] if contrib.nnz else []
        why = ("Correspond à votre recherche : " + ", ".join(terms)) if terms else "Article populaire (aucun mot-clé en commun)"
        rows.append(_row(i, r + 1, s[i], why))
    return pd.DataFrame(rows), "**Cold-start total** : aucun historique → contenu (80 %) + popularité (20 %). " + msg


with gr.Blocks(title="Recommandeur hybride Amazon") as demo:
    gr.Markdown("# Recommandeur hybride – Amazon Reviews 2023\n"
                "Hybride **pondéré + switching** (CF item-item, CF SVD, contenu TF-IDF, popularité) avec filtres par contraintes et explications.")
    with gr.Row():
        max_price = gr.Number(label="Prix max ($) — optionnel", value=None)
        min_rating = gr.Slider(0, 5, value=0, step=0.5, label="Note moyenne minimale")
        category = gr.Dropdown(CATS, value="Toutes", label="Catégorie")
        k = gr.Slider(3, 20, value=10, step=1, label="Nombre de recommandations")
    with gr.Tabs():
        with gr.Tab("Utilisateur existant"):
            user = gr.Dropdown(user_choices, value=user_choices[0][1], label="Utilisateur")
            with gr.Row():
                simulate = gr.Checkbox(label="Simuler un NOUVEL utilisateur (cold-start)", value=False)
                n_keep = gr.Slider(1, 5, value=2, step=1, label="…en ne gardant que ses N premières interactions")
            btn = gr.Button("Recommander", variant="primary")
            info = gr.Markdown()
            gr.Markdown("### Historique (15 dernières interactions visibles)")
            hist_out = gr.Dataframe(wrap=True)
            gr.Markdown("### Recommandations")
            rec_out = gr.Dataframe(wrap=True)
            btn.click(recommend_existing, [user, k, simulate, n_keep, max_price, min_rating, category], [hist_out, rec_out, info])
        with gr.Tab("Nouvel utilisateur (mots-clés)"):
            q = gr.Textbox(label="Que cherchez-vous ?", placeholder="ex. guitar strings")
            btn2 = gr.Button("Recommander", variant="primary")
            info2 = gr.Markdown(); rec2 = gr.Dataframe(wrap=True)
            btn2.click(recommend_new_user, [q, k, max_price, min_rating, category], [rec2, info2])

if __name__ == "__main__":
    demo.launch()
