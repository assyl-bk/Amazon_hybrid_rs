"""Génère un MINI-JEU DE DONNÉES SYNTHÉTIQUE au même format qu'Amazon Reviews 2023.
UNIQUEMENT pour tester que le code tourne (CI, debug). Les résultats obtenus dessus n'ont AUCUNE valeur
scientifique : ne jamais les mettre dans le rapport.

Usage :  RS_CATEGORIES=Synth_A,Synth_B,Synth_C RS_USERS_PER_CAT=400 python scripts/make_synthetic.py
Format : Synth_A au format '5core CSV' (train/valid/test), Synth_B et Synth_C au format 'brut jsonl.gz'.
"""
import gzip, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

rng = np.random.default_rng(0)
TOPICS = {t: [f"{t}{j}" for j in range(25)] for t in ["guitar", "piano", "drums", "rpg", "racing", "puzzle", "antivirus", "office", "photo"]}
COMMON = ["great", "quality", "new", "classic", "edition", "pack", "premium", "portable"]
CAT_TOPICS = {0: ["guitar", "piano", "drums"], 1: ["rpg", "racing", "puzzle"], 2: ["antivirus", "office", "photo"]}
N_ITEMS, USERS = 700, int(C.N_USERS_PER_CAT * 1.3)

C.DATA_RAW.mkdir(parents=True, exist_ok=True)
for c, cat in enumerate(C.CATEGORIES[:3]):
    topics = CAT_TOPICS[c]
    it_topic = rng.integers(0, 3, N_ITEMS)
    it_pop = rng.zipf(1.6, N_ITEMS).astype(float); it_pop = np.minimum(it_pop, 60)
    ids = [f"{cat[-1].upper()}{k:06d}" for k in range(N_ITEMS)]
    with gzip.open(C.DATA_RAW / f"meta_{cat}.jsonl.gz", "wt") as f:
        for k in range(N_ITEMS):
            words = list(rng.choice(TOPICS[topics[it_topic[k]]], 6)) + list(rng.choice(COMMON, 2))
            f.write(json.dumps({"parent_asin": ids[k], "title": f"{topics[it_topic[k]].title()} item {k} " + " ".join(words[:3]),
                                "features": [" ".join(words[:4])], "description": [" ".join(words)],
                                "categories": [cat, topics[it_topic[k]]], "price": str(round(float(rng.uniform(5, 200)), 2)),
                                "average_rating": round(float(rng.uniform(3, 5)), 1), "rating_number": int(it_pop[k] * 5),
                                "store": f"Brand{rng.integers(0, 30)}"}) + "\n")
    rows = []
    for u in range(USERS):
        fav = rng.choice(3, size=int(rng.integers(1, 3)), replace=False)
        n = int(rng.integers(6, 22))
        w = it_pop * np.where(np.isin(it_topic, fav), 8.0, 1.0); w /= w.sum()
        items = rng.choice(N_ITEMS, size=min(n, N_ITEMS), replace=False, p=w)
        t0 = 1_400_000_000 + int(rng.integers(0, 10**8))
        for j, it in enumerate(items):
            match = it_topic[it] in fav
            r = int(rng.choice([4, 5], p=[.4, .6])) if match else int(rng.choice([1, 2, 3, 4, 5], p=[.1, .1, .2, .3, .3]))
            rows.append((f"U{cat[-1]}{u:05d}", ids[it], float(r), (t0 + j * 86400 * int(rng.integers(1, 30))) * 1000))
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])
    if c == 0:   # format 5-core CSV (leave-one-out)
        df = df.sort_values(["user_id", "timestamp"])
        rk = df.groupby("user_id").cumcount(ascending=False)
        df[rk >= 2].to_csv(C.DATA_RAW / f"{cat}.train.csv", index=False)
        df[rk == 1].to_csv(C.DATA_RAW / f"{cat}.valid.csv", index=False)
        df[rk == 0].to_csv(C.DATA_RAW / f"{cat}.test.csv", index=False)
    else:        # format brut jsonl.gz
        with gzip.open(C.DATA_RAW / f"{cat}.jsonl.gz", "wt") as f:
            for r in df.itertuples():
                f.write(json.dumps({"user_id": r.user_id, "parent_asin": r.parent_asin, "rating": r.rating, "timestamp": r.timestamp}) + "\n")
print("Données synthétiques écrites dans", C.DATA_RAW)
