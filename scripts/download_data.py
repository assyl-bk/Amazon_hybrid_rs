"""Télécharge GRATUITEMENT (sans compte) les données Amazon Reviews 2023 (McAuley Lab, UCSD) dans data/raw/.

Deux modes :
  --mode 5core  (défaut, LÉGER) : interactions pré-filtrées 5-core depuis Hugging Face (petits CSV)
                                  + métadonnées produits (jsonl.gz) depuis datarepo.eng.ucsd.edu
  --mode raw    (LOURD)         : avis bruts (.jsonl) + métadonnées ; on applique alors notre propre k-core.

Exemples :
  python scripts/download_data.py
  python scripts/download_data.py --cats Video_Games Software Musical_Instruments
  python scripts/download_data.py --mode raw --cats Digital_Music Software

Catégories disponibles : voir https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
"""
import argparse, sys
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core"
HF_SINGLE = HF_BASE + "/rating_only/{cat}.csv"            # 1 fichier : user_id,parent_asin,rating,timestamp
HF_SPLIT = HF_BASE + "/last_out/{cat}.{split}.csv"        # repli : leave-one-out train/valid/test
# Les fichiers bruts sont hébergés sur Hugging Face en .jsonl NON compressé (l'ancien miroir UCSD .jsonl.gz renvoie 404).
HF_RAW = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/{kind}_categories/{prefix}{cat}.jsonl"
UCSD = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/{kind}_categories/{prefix}{cat}.jsonl.gz"  # repli


def download(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [déjà là] {dest.name}"); return True
    print(f"  ↓ {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)); done = 0
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk); done += len(chunk)
                    if total: print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} Mo", end="")
            tmp.rename(dest); print()
        return True
    except Exception as e:
        print(f"\n  ✗ échec : {e}")
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="+", default=C.CATEGORIES)
    ap.add_argument("--mode", choices=["5core", "raw"], default="5core")
    a = ap.parse_args()
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    for cat in a.cats:
        print(f"== {cat}")
        ok = True
        if a.mode == "5core":
            ok = download(HF_SINGLE.format(cat=cat), C.DATA_RAW / f"{cat}.csv")
            if not ok:
                print("  -> repli sur les fichiers last_out (train/valid/test)")
                ok = True
                for s in ("train", "valid", "test"):
                    ok &= download(HF_SPLIT.format(cat=cat, split=s), C.DATA_RAW / f"{cat}.{s}.csv")
            if not ok:
                print("  -> CSV 5-core introuvable : télécharge-le à la main (voir README) ou relance avec --mode raw")
        else:
            ok &= (download(HF_RAW.format(kind="review", prefix="", cat=cat), C.DATA_RAW / f"{cat}.jsonl")
                   or download(UCSD.format(kind="review", prefix="", cat=cat), C.DATA_RAW / f"{cat}.jsonl.gz"))
        ok &= (download(HF_RAW.format(kind="meta", prefix="meta_", cat=cat), C.DATA_RAW / f"meta_{cat}.jsonl")
               or download(UCSD.format(kind="meta", prefix="meta_", cat=cat), C.DATA_RAW / f"meta_{cat}.jsonl.gz"))
    print("\nTerminé. Étape suivante : ouvre notebooks/projet_rs_hybride.ipynb")