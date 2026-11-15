"""Exécute tout le pipeline en ligne de commande (sans notebook) : données -> modèles -> tuning -> résultats -> artefacts."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import data as D, pipeline as P
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

ds = D.build_dataset()
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in D.basic_stats(ds).items()})
res = P.run_experiments(ds, verbose=True)
for n, t in res["cold_users"].items():
    print(f"\n--- Cold-start utilisateurs ({n} interaction(s) d'historique) ---"); print(t.round(4))
print(f"\n--- Cold-start items ({res['cold_items_n_users']} utilisateurs) ---"); print(res["cold_items"].round(4))
print("\nArtefacts sauvegardés dans artifacts/ -> lance : python app.py")
