"""Modèles de recommandation : popularité, item-kNN (CF mémoire), SVD (CF modèle), contenu TF-IDF, hybrides.

Interface commune : model.fit(R) puis model.score(H) -> matrice (n_users_batch x n_items)
 - R : matrice binaire utilisateurs x items (apprentissage)
 - H : matrice binaire (batch d'utilisateurs) x items = HISTORIQUE des utilisateurs à servir.
Scorer à partir de H (et non d'un index utilisateur) permet de servir des utilisateurs
quasi-nouveaux (fold-in) sans réentraîner : indispensable pour l'étude du cold-start.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import config as C


def row_max_normalize(S: np.ndarray) -> np.ndarray:
    m = S.max(axis=1, keepdims=True)
    m[m <= 0] = 1.0
    return S / m


class Popularity:
    name = "Popularité"
    def fit(self, R, **kw):
        self.pop = np.asarray(R.sum(axis=0)).ravel().astype(np.float64)
        self.s = np.log1p(self.pop)
        return self
    def score(self, H):
        return np.tile(self.s, (H.shape[0], 1))


class ItemKNN:
    """Filtrage collaboratif item-item (cosinus), k plus proches voisins par item (chapitre 2)."""
    name = "CF item-item"
    def __init__(self, k=C.K_NEIGHBORS, chunk=1000):
        self.k, self.chunk = k, chunk
    def fit(self, R, **kw):
        R = R.tocsc().astype(np.float32)
        n_items = R.shape[1]
        norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=0)).ravel()); norms[norms == 0] = 1
        rows, cols, vals = [], [], []
        Rt = R.T.tocsr()
        for s in range(0, n_items, self.chunk):
            e = min(s + self.chunk, n_items)
            sim = (Rt[s:e] @ R).toarray()                    # (chunk x n_items) co-occurrences
            sim /= norms[s:e, None]; sim /= norms[None, :]
            sim[np.arange(e - s), np.arange(s, e)] = 0       # pas d'auto-similarité
            k = min(self.k, n_items - 1)
            idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            v = np.take_along_axis(sim, idx, axis=1)
            keep = v > 0
            rr = np.repeat(np.arange(s, e), k).reshape(-1, k)
            rows.append(rr[keep]); cols.append(idx[keep]); vals.append(v[keep])
        self.S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(n_items, n_items), dtype=np.float32)
        return self
    def score(self, H):
        return (H.astype(np.float32) @ self.S).toarray()


class SVDModel:
    """PureSVD : factorisation de la matrice binaire ; l'utilisateur est projeté (fold-in) dans l'espace latent."""
    name = "CF SVD"
    def __init__(self, dim=C.SVD_DIM):
        self.dim = dim
    def fit(self, R, **kw):
        dim = min(self.dim, min(R.shape) - 1)
        self.svd = TruncatedSVD(n_components=dim, random_state=C.SEED).fit(R)
        self.V = self.svd.components_.T.astype(np.float32)     # (n_items x dim)
        return self
    def score(self, H):
        return ((H.astype(np.float32) @ self.V) @ self.V.T)


class ContentTFIDF:
    """Approche basée sur le contenu : TF-IDF sur (titre + catégories + caractéristiques + description) (chapitre 3)."""
    name = "Contenu TF-IDF"
    def __init__(self, texts: list[str] | None = None):
        self.texts = texts
    def fit(self, R=None, texts=None, **kw):
        texts = texts if texts is not None else self.texts
        self.vec = TfidfVectorizer(stop_words="english", max_features=C.TFIDF_MAX_FEATURES, min_df=2,
                                   sublinear_tf=True, ngram_range=(1, 2), dtype=np.float32)
        self.X = self.vec.fit_transform(texts).tocsr()        # lignes L2-normalisées
        self.XT = self.X.T.tocsr()
        return self
    def profile(self, H):
        Hn = normalize(H.astype(np.float32), norm="l1")       # moyenne des vecteurs des items de l'historique
        return normalize(Hn @ self.X)
    def score(self, H):
        return (self.profile(H) @ self.XT).toarray()
    def score_query(self, text: str):
        q = self.vec.transform([text])
        return (q @ self.XT).toarray()


class HybridWeighted:
    """Hybridation PONDÉRÉE (chapitre 5) : somme pondérée des scores normalisés de plusieurs moteurs.
    item_fallback=True : les items SANS aucune interaction d'apprentissage (nouveaux items) ne peuvent pas
    être scorés par le CF ; ils sont alors scorés par le contenu seul (commutation au niveau de l'item)."""
    name = "Hybride pondéré"
    def __init__(self, models: dict, weights: dict, item_fallback=False):
        self.models, self.weights, self.item_fallback = models, weights, item_fallback
    def components(self, H):
        return {k: row_max_normalize(m.score(H)) for k, m in self.models.items() if self.weights.get(k, 0) > 0}
    def score(self, H):
        comp = self.components(H)
        S = sum(self.weights[k] * v for k, v in comp.items())
        if self.item_fallback:
            new_items = self.models["pop"].pop == 0
            if new_items.any():
                c = comp["content"] if "content" in comp else row_max_normalize(self.models["content"].score(H))
                S[:, new_items] = c[:, new_items]
        return S


class HybridSwitch:
    """Hybridation par COMMUTATION + PONDÉRATION (chapitre 5), deux niveaux de commutation :
      - utilisateur : jeu de poids 'froid' si l'historique est court (< T interactions), 'chaud' sinon ;
      - item : un item sans interaction est scoré par le contenu seul."""
    name = "Hybride switching"
    def __init__(self, models: dict, w_cold: dict, w_warm: dict, T=C.COLD_T):
        self.models, self.w_cold, self.w_warm, self.T = models, w_cold, w_warm, T
        self.cold = HybridWeighted(models, w_cold, item_fallback=True)
        self.warm = HybridWeighted(models, w_warm, item_fallback=True)
    def is_cold(self, H):
        return np.asarray(H.sum(axis=1)).ravel() < self.T
    def weights_for(self, n_hist):
        return self.w_cold if n_hist < self.T else self.w_warm
    def score(self, H):
        cold = self.is_cold(H)
        out = np.zeros((H.shape[0], self.models["content"].X.shape[0]))
        if (~cold).any():
            out[~cold] = self.warm.score(H[~cold])
        if cold.any():
            out[cold] = self.cold.score(H[cold])
        return out


def fit_all(R, texts):
    """Entraîne les 4 modèles de base sur la matrice R. Retourne un dict {clé: modèle}."""
    return {"pop": Popularity().fit(R), "knn": ItemKNN().fit(R), "svd": SVDModel().fit(R),
            "content": ContentTFIDF().fit(texts=texts)}
