"""Partie 2 : profils de logements par clustering (Fabrice Robert).

Reprend la démarche du notebook `02_clustering_logements_ames.ipynb` :
KMeans à 9 groupes sur 12 caractéristiques physiques standardisées (sans le prix),
puis nommage automatique de chaque groupe selon sa taille, son âge et sa qualité moyens.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import MODELS_DIR, RANDOM_STATE, load_dataset

CLUSTER_FEATURES = ["Total_SF", "Lot_Area", "House_Age", "Years_Since_Remod",
                    "Overall_Qual", "Overall_Cond", "Bedroom_AbvGr", "TotRms_AbvGrd",
                    "Total_Bathrooms", "Garage_Cars", "Garage_Area", "Fireplaces"]
K = 9


def tertile_labels(series: pd.Series, labels: list) -> pd.Series:
    return pd.qcut(series.rank(method="first"), 3, labels=labels)


def main() -> dict:
    df = load_dataset()
    X = df[CLUSTER_FEATURES]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    silhouettes = {}
    for k in range(2, 11):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(Xs)
        silhouettes[k] = round(float(silhouette_score(Xs, labels)), 4)

    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    df["Cluster"] = kmeans.fit_predict(Xs)

    # comme dans le notebook d'origine, le nommage se fait sur les moyennes arrondies à 0,1
    profile = df.groupby("Cluster")[CLUSTER_FEATURES].mean().round(1)
    size = tertile_labels(profile["Total_SF"], ["petit", "moyen", "grand"])
    age = tertile_labels(profile["House_Age"], ["récent", "d'âge intermédiaire", "ancien"])
    qual = tertile_labels(profile["Overall_Qual"], ["entrée de gamme", "milieu de gamme", "haut de gamme"])
    label_map = (size.astype(str).str.capitalize() + " logement " + age.astype(str)
                 + " (" + qual.astype(str) + ")").to_dict()
    df["Type_Logement"] = df["Cluster"].map(label_map)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Xs)

    summary = df.groupby("Cluster").agg(
        type_logement=("Type_Logement", "first"), nb_logements=("Cluster", "size"),
        surface_totale_pieds2=("Total_SF", "mean"), age_moyen=("House_Age", "mean"),
        qualite_moyenne=("Overall_Qual", "mean"), prix_median=("SalePrice", "median"),
    ).round(1).reset_index()

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "kmeans": kmeans, "pca": pca,
                 "features": CLUSTER_FEATURES, "label_map": label_map},
                MODELS_DIR / "clustering.joblib")
    pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": df["Cluster"],
                  "type_logement": df["Type_Logement"], "prix": df["SalePrice"]}
                 ).to_csv(MODELS_DIR / "clustering_points.csv", index=False)
    summary.to_csv(MODELS_DIR / "clustering_profils.csv", index=False)
    df[["Cluster", "Type_Logement"]].to_csv(MODELS_DIR / "clustering_labels.csv", index=False)

    metrics = {"k": K, "silhouette_par_k": silhouettes, "silhouette_k9": silhouettes[K],
               "variance_expliquee_pca": [round(float(v), 3) for v in pca.explained_variance_ratio_]}
    (MODELS_DIR / "clustering_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))
    print("Silhouette k=9 :", silhouettes[K])
    return metrics


if __name__ == "__main__":
    main()
