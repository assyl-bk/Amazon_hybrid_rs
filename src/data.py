"""Chargement, nettoyage, échantillonnage équilibré et découpage temporel des données Amazon Reviews 2023."""
from __future__ import annotations
import gzip, json
from dataclasses import dataclass
import numpy as np
import pandas as pd
import scipy.sparse as sp
import config as C


# ----------------------------------------------------------------------------- lecture
def _find(name: str):
    """Retourne le chemin de `name` (.jsonl.gz ou .jsonl) s'il existe dans data/raw, sinon None."""
    for ext in (".jsonl.gz", ".jsonl"):
        p = C.DATA_RAW / f"{name}{ext}"
        if p.exists():
            return p
    return None


def _open_text(p):
    return gzip.open(p, "rt", encoding="utf-8") if str(p).endswith(".gz") else open(p, "r", encoding="utf-8")


def _fix_ts(ts: pd.Series) -> pd.Series:
    ts = pd.to_numeric(ts, errors="coerce")
    return ts.where(ts < 1e11, ts / 1000.0)  # ms -> s


def _read_5core_csv(cat: str) -> pd.DataFrame | None:
    """CSV 5-core du benchmark officiel : soit un seul fichier 'rating_only' <cat>.csv,
    soit les 3 fichiers <cat>.train/valid/test.csv (on les recolle, notre code refait son propre split)."""
    single = C.DATA_RAW / f"{cat}.csv"
    if single.exists():
        df = pd.read_csv(single)
    else:
        parts = []
        for s in ("train", "valid", "test"):
            p = C.DATA_RAW / f"{cat}.{s}.csv"
            if not p.exists():
                return None
            parts.append(pd.read_csv(p))
        df = pd.concat(parts, ignore_index=True)
    if "parent_asin" not in df.columns and "asin" in df.columns:
        df = df.rename(columns={"asin": "parent_asin"})
    return df[["user_id", "parent_asin", "rating", "timestamp"]]


def _read_raw_reviews(cat: str) -> pd.DataFrame | None:
    p = _find(cat)
    if p is None:
        return None
    rows = []
    with _open_text(p) as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["user_id"], d["parent_asin"], d["rating"], d.get("timestamp", d.get("sort_timestamp"))))
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])


def load_interactions(cats=None) -> pd.DataFrame:
    cats = cats or C.CATEGORIES
    out = []
    for cat in cats:
        df = _read_5core_csv(cat)
        if df is None:
            df = _read_raw_reviews(cat)
        if df is None:
            raise FileNotFoundError(
                f"Aucune donnée pour '{cat}' dans {C.DATA_RAW}. Lance d'abord : python scripts/download_data.py")
        df = df.rename(columns={"parent_asin": "item_id"})
        df["category"] = cat
        out.append(df)
    df = pd.concat(out, ignore_index=True)
    df["timestamp"] = _fix_ts(df["timestamp"])
    return df.dropna(subset=["user_id", "item_id", "rating", "timestamp"])


def _flatten(x, limit=1500) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        x = " ".join(str(v) for v in x if v)
    return str(x)[:limit]


def load_meta(cats, keep_items: set) -> pd.DataFrame:
    rows = []
    for cat in cats:
        p = _find(f"meta_{cat}")
        if p is None:
            raise FileNotFoundError(f"Métadonnées manquantes : data/raw/meta_{cat}.jsonl (ou .jsonl.gz). "
                                    "Lance scripts/download_data.py ou télécharge-les à la main (voir README).")
        with _open_text(p) as f:
            for line in f:
                d = json.loads(line)
                iid = d.get("parent_asin")
                if iid not in keep_items:
                    continue
                price = pd.to_numeric(d.get("price"), errors="coerce")
                cats_path = d.get("categories") or []
                rows.append({
                    "item_id": iid,
                    "title": d.get("title") or "",
                    "text": " ".join([d.get("title") or "", _flatten(cats_path), _flatten(d.get("features")),
                                      _flatten(d.get("description")), d.get("store") or ""]),
                    "price": price,
                    "avg_rating": d.get("average_rating"),
                    "n_ratings": d.get("rating_number"),
                    "meta_category": cat,
                })
    meta = pd.DataFrame(rows).drop_duplicates("item_id")
    return meta


# ----------------------------------------------------------------------------- nettoyage / équilibrage
def iterative_core(df: pd.DataFrame, k_user: int, k_item: int) -> pd.DataFrame:
    while True:
        n0 = len(df)
        uc = df.groupby("user_id")["item_id"].transform("size")
        df = df[uc >= k_user]
        ic = df.groupby("item_id")["user_id"].transform("size")
        df = df[ic >= k_item]
        if len(df) == n0:
            return df


def balanced_sample(df: pd.DataFrame, n_per_cat: int, seed: int) -> pd.DataFrame:
    """Chaque utilisateur est rattaché à sa catégorie dominante ; on tire le même nombre
    d'utilisateurs par catégorie -> catalogue équilibré entre catégories."""
    rng = np.random.default_rng(seed)
    main = (df.groupby(["user_id", "category"]).size().reset_index(name="n")
              .sort_values(["user_id", "n"], ascending=[True, False]).drop_duplicates("user_id")
              .set_index("user_id")["category"])
    keep = []
    for _, users in main.groupby(main).groups.items():
        users = np.array(list(users))
        keep += list(rng.choice(users, min(n_per_cat, len(users)), replace=False))
    return df[df["user_id"].isin(set(keep))]


@dataclass
class Dataset:
    inter: pd.DataFrame        # user_id, item_id, rating, timestamp, category, u, i
    items: pd.DataFrame        # indexé par i (0..n_items-1) : item_id, title, text, price, avg_rating, category...
    n_users: int
    n_items: int
    user_main_cat: pd.Series   # u -> catégorie dominante


def build_dataset(cats=None, n_per_cat=None, verbose=True) -> Dataset:
    cats = cats or C.CATEGORIES
    n_per_cat = n_per_cat or C.N_USERS_PER_CAT
    df = load_interactions(cats)
    df = df.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
    if verbose:
        print(f"Brut : {len(df):,} interactions, {df.user_id.nunique():,} utilisateurs, {df.item_id.nunique():,} items")
    df = df[df["rating"] >= C.POS_THRESHOLD]
    df = iterative_core(df, C.MIN_USER_INTER, C.MIN_ITEM_INTER)
    if verbose:
        print(f"Positifs (note>={C.POS_THRESHOLD}) + k-core : {len(df):,} interactions, {df.user_id.nunique():,} utilisateurs")
    df = balanced_sample(df, n_per_cat, C.SEED)
    df = iterative_core(df, C.MIN_USER_INTER, C.MIN_ITEM_INTER)
    meta = load_meta(cats, set(df["item_id"]))
    # on ne garde que les items qui ont des métadonnées (nécessaire pour l'approche contenu + l'interface)
    df = df[df["item_id"].isin(set(meta["item_id"]))]
    df = iterative_core(df, C.MIN_USER_INTER, C.MIN_ITEM_INTER)
    meta = meta[meta["item_id"].isin(set(df["item_id"]))]

    users = np.sort(df["user_id"].unique()); items = np.sort(df["item_id"].unique())
    u_map = {u: k for k, u in enumerate(users)}; i_map = {i: k for k, i in enumerate(items)}
    df = df.assign(u=df["user_id"].map(u_map), i=df["item_id"].map(i_map)).sort_values(["u", "timestamp"]).reset_index(drop=True)
    meta = meta.assign(i=meta["item_id"].map(i_map)).sort_values("i").set_index("i", drop=False)
    # catégorie de l'item = catégorie du fichier d'où vient son interaction majoritaire
    meta["category"] = df.groupby("i")["category"].agg(lambda s: s.mode().iloc[0]).reindex(meta.index)
    main = df.groupby("u")["category"].agg(lambda s: s.mode().iloc[0])
    ds = Dataset(df, meta, len(users), len(items), main)
    if verbose:
        print(f"Final : {len(df):,} interactions, {ds.n_users:,} utilisateurs, {ds.n_items:,} items")
    return ds


# ----------------------------------------------------------------------------- découpage temporel
def temporal_split(inter: pd.DataFrame):
    """Leave-one-out temporel : dernière interaction -> test, avant-dernière -> validation, reste -> train."""
    rk = inter.groupby("u").cumcount(ascending=False)
    return inter[rk >= 2].copy(), inter[rk == 1].copy(), inter[rk == 0].copy()


def to_csr(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    return sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df["u"].values, df["i"].values)),
                         shape=(n_users, n_items))


def basic_stats(ds: Dataset) -> dict:
    nU, nI, nR = ds.n_users, ds.n_items, len(ds.inter)
    pop = ds.inter.groupby("i").size().to_numpy(dtype=np.int64)
    top1 = max(1, int(0.01 * nI))
    return {"utilisateurs": nU, "items": nI, "interactions": nR,
            "sparsité": 1 - nR / (nU * nI),
            "interactions/utilisateur (médiane)": float(ds.inter.groupby("u").size().median()),
            "interactions/item (médiane)": float(np.median(pop)),
            "part des interactions captée par le top 1% des items": float(pop[:top1].sum() / nR)}