"""Vérifie nos métriques contre les valeurs calculées à la main dans le cours (chapitre 6).
Lancer :  python tests/test_metrics.py"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics as M

rel = {"i1", "i3", "i7"}
L = ["i5", "i1", "i3", "i7", "i9"]
Mm = ["i1", "i5", "i9", "i3", "i7"]

def close(a, b, tol=0.01): assert abs(a - b) < tol, (a, b)

close(M.precision_at_k(L, rel, 3), 2 / 3); close(M.precision_at_k(Mm, rel, 3), 1 / 3)
close(M.recall_at_k(L, rel, 3), 2 / 3);    close(M.recall_at_k(Mm, rel, 3), 1 / 3)
close(M.precision_at_k(L, rel, 5), 0.6);   close(M.precision_at_k(Mm, rel, 5), 0.6)
close(M.average_precision(L, rel), 0.64);  close(M.average_precision(Mm, rel), 0.70)
close(M.ndcg_at_k(L, rel, 5), 0.73);       close(M.ndcg_at_k(Mm, rel, 5), 0.85)
close(M.reciprocal_rank(L, rel), 0.5)
assert M.hit_at_k(L, rel, 3) == 1.0
close(M.coverage([[0, 1], [1, 2]], 6), 0.5)
print("OK : toutes les métriques correspondent aux exemples du cours.")
