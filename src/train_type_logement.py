"""Partie 3 : classification du type de logement (Duncan Costes).

Reprend la démarche du notebook `classification_type_logement.ipynb` :
on apprend à prédire le `Type_Logement` issu du clustering à partir de toutes les
caractéristiques du bien (le numéro de cluster est retiré pour éviter la fuite de données),
3 modèles sont comparés et la régression logistique est retenue.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from common import MODELS_DIR, RANDOM_STATE, load_dataset

MIN_SAMPLES = 5
# le notebook d'origine travaille sur le jeu exporté par le clustering :
# colonnes brutes + 4 variables dérivées (sans les indicateurs Has_*)
EXCLUDED = ["Has_Pool", "Has_Garage", "Has_Fireplace"]


def main() -> dict:
    df = load_dataset().drop(columns=EXCLUDED)
    labels = pd.read_csv(MODELS_DIR / "clustering_labels.csv")
    df["Type_Logement"] = labels["Type_Logement"].values

    counts = df["Type_Logement"].value_counts()
    rares = counts[counts < MIN_SAMPLES].index.tolist()
    df = df[~df["Type_Logement"].isin(rares)]

    X = df.drop(columns=["Type_Logement"])
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(exclude=[np.number]).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="None")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    le = LabelEncoder()
    y = le.fit_transform(df["Type_Logement"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    models = {
        "Régression logistique": LogisticRegression(max_iter=1000, class_weight="balanced",
                                                    random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    comparison, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("model", model)])
        s = cross_validate(pipe, X_train, y_train, cv=5, scoring=["accuracy", "f1_macro"], n_jobs=-1)
        fitted[name] = pipe.fit(X_train, y_train)
        comparison.append({"modele": name, "accuracy_cv": round(float(s["test_accuracy"].mean()), 4),
                           "f1_macro_cv": round(float(s["test_f1_macro"].mean()), 4)})
        print(f"{name:22s} accuracy CV = {s['test_accuracy'].mean():.3f}  f1 = {s['test_f1_macro'].mean():.3f}")

    best = max(comparison, key=lambda r: r["f1_macro_cv"])["modele"]
    pipe = fitted[best]
    pred = pipe.predict(X_test)
    metrics = {"modele_retenu": best, "accuracy_test": round(float(accuracy_score(y_test, pred)), 4),
               "f1_macro_test": round(float(f1_score(y_test, pred, average="macro")), 4),
               "classes_retirees": rares, "comparaison_cv": comparison}
    print(json.dumps({k: v for k, v in metrics.items() if k != "comparaison_cv"}, ensure_ascii=False))

    joblib.dump({"pipeline": pipe, "label_encoder": le, "columns": X.columns.tolist()},
                MODELS_DIR / "type_logement.joblib")
    (MODELS_DIR / "type_logement_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    main()
