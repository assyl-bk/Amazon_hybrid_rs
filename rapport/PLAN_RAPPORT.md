# Plan du rapport (10 pages maximum) — à rédiger avec TES résultats

1. **Introduction (0,5 p.)** — problème, domaine (e-commerce), dataset Amazon Reviews 2023, objectifs.
2. **Données (1,5 p.)** — catégories choisies, filtrage (note ≥ 4, k-core, échantillon équilibré), tableau de statistiques
   (utilisateurs, items, interactions, sparsité), distribution des notes, longue traîne, split leave-one-out temporel.
3. **Approches de base (1,5 p.)** — popularité, item-kNN, SVD, TF-IDF : principe, paramètres (K voisins, dimension, TF-IDF), justification.
4. **Approche hybride (1,5 p.)** — stratégie pondérée + switching (utilisateur et item), normalisation, grille de poids, schéma d'architecture,
   lien avec la taxonomie du chapitre 5, filtre par contraintes (chapitre 4).
5. **Protocole d'évaluation (1 p.)** — classement complet, métriques (P/R/HR/NDCG/MRR, couverture, diversité, nouveauté), cold-start utilisateurs
   (historique tronqué à 1/2/3) et items (5 % retirés), séparation tuning/test, limites du leave-one-out (Recall@K = HR@K).
6. **Résultats et discussion (2,5 p.)** — tableaux + figures : global, cold-start utilisateurs, cold-start items, au-delà de la précision.
   Analyse critique : l'hybride gagne-t-il vraiment ? incertitude statistique ? compromis précision/diversité ?
7. **Démonstration (0,5 p.)** — captures de l'interface, exemples d'explications.
8. **Utilisation de l'IA générative (0,5 p. — obligatoire)** — parties générées / écrites à la main, erreurs de l'IA repérées et corrigées,
   choix techniques personnels.
9. **Conclusion et perspectives (0,5 p.)** — limites, pistes (embeddings de phrases, LightGCN, bandits, explicabilité…).
10. **Références** — Hou et al. 2024 (Amazon Reviews 2023), cours (chap. 1–7).
