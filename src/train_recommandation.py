"""Partie 4 : système de recommandation (Abdellatif Essid).

Reprend la démarche du notebook `Système_de_Recommandation_Immobilier.ipynb` :
un modèle d'estimation léger (8 caractéristiques) est entraîné sur 80 % des maisons ;
les 20 % restantes forment un « catalogue » de biens en vente. Pour chacune, on compare
son prix réel à l'estimation du modèle (Deal Index) afin de repérer les bonnes affaires.

Adaptation : le notebook d'origine utilisait la version Kaggle du jeu de données
(1 460 maisons) ; ici, on utilise le même jeu complet que les autres parties.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from common import MODELS_DIR, RANDOM_STATE, load_dataset

FEATURES_NUM = ["Gr_Liv_Area", "Total_Bsmt_SF", "Bedroom_AbvGr", "Garage_Cars",
                "Full_Bath", "Year_Built", "Overall_Qual"]
FEATURES_CAT = ["Neighborhood"]


def main() -> dict:
    df = load_dataset()
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df["SalePrice"]
    X_train, X_cat, y_train, y_cat = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
         FEATURES_NUM),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), FEATURES_CAT),
    ])
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=RANDOM_STATE),
    }
    best, best_rmse, comparison = None, np.inf, []
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", pre), ("regressor", model)]).fit(X_train, y_train)
        pred = pipe.predict(X_cat)
        rmse = float(np.sqrt(mean_squared_error(y_cat, pred)))
        comparison.append({"modele": name, "rmse_dollars": round(rmse, 0),
                           "r2": round(float(r2_score(y_cat, pred)), 4)})
        print(f"{name:14s} RMSE = {rmse:,.0f} $  R² = {r2_score(y_cat, pred):.4f}")
        if rmse < best_rmse:
            best, best_rmse, best_name = pipe, rmse, name

    catalogue = df.loc[X_cat.index, FEATURES_NUM + FEATURES_CAT
                       + ["Total_SF", "Total_Bathrooms", "Sale_Condition"]].copy()
    catalogue["Prix"] = y_cat
    catalogue["Estimation"] = best.predict(X_cat)
    catalogue["DealIndex"] = (catalogue["Estimation"] - catalogue["Prix"]) / catalogue["Prix"] * 100
    catalogue.index.name = "ID"
    catalogue.reset_index().to_csv(MODELS_DIR / "recommandation_catalogue.csv", index=False)

    metrics = {"modele_retenu": best_name, "rmse_dollars": round(best_rmse, 0),
               "taille_catalogue": int(len(catalogue)), "comparaison": comparison}
    joblib.dump(best, MODELS_DIR / "recommandation_estimateur.joblib")
    (MODELS_DIR / "recommandation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def recommander(catalogue: pd.DataFrame, budget_max: float, min_chambres: int,
                min_garage: int, min_sdb: int = 1, n: int = 3,
                ventes_normales: bool = True) -> pd.DataFrame:
    """Filtres stricts du client, puis classement par qualité × 10 + Deal Index (méthode d'origine).

    `ventes_normales` (ajout) écarte les ventes inhabituelles (entre proches, saisies, maisons
    pas encore terminées…), dont le prix ne reflète pas le marché et gonfle artificiellement
    le Deal Index — un doute déjà relevé dans le notebook d'origine.
    """
    ok = catalogue[(catalogue["Prix"] <= budget_max) & (catalogue["Bedroom_AbvGr"] >= min_chambres)
                   & (catalogue["Garage_Cars"] >= min_garage) & (catalogue["Full_Bath"] >= min_sdb)].copy()
    if ventes_normales:
        ok = ok[ok["Sale_Condition"] == "Normal"]
    ok["Score"] = ok["Overall_Qual"] * 10 + ok["DealIndex"]
    return ok.sort_values("Score", ascending=False).head(n)


if __name__ == "__main__":
    main()
