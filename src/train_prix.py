"""Partie 1 : estimation du prix de vente (Yannis Bouttier).

Reprend la démarche du notebook `02_prediction_prix_ames.ipynb` :
5 modèles comparés en validation croisée (5 plis) puis XGBoost retenu,
entraîné sur log(1 + prix) et évalué en dollars sur 20 % de maisons jamais vues.

Par défaut, les hyperparamètres sont ceux trouvés par la recherche en grille du notebook.
Lancer avec `--grid` pour refaire la recherche complète (plus long).
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from common import MODELS_DIR, RANDOM_STATE, load_dataset

# résultat de la recherche en grille (`--grid`) avec les versions de requirements.txt
# (le notebook d'origine, avec des versions plus anciennes, avait trouvé n_estimators=800)
BEST_PARAMS = {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.1}


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def main(grid: bool = False) -> dict:
    df = load_dataset()
    X = df.drop(columns=["SalePrice"])
    y = np.log1p(df["SalePrice"])
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X, y, df["SalePrice"], test_size=0.2, random_state=RANDOM_STATE)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "Baseline (moyenne)": DummyRegressor(strategy="mean"),
        "Régression linéaire": LinearRegression(),
        "Ridge (L2)": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Lasso (L1)": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=10000),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8,
                                colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1),
    }
    comparison = []
    for name, model in candidates.items():
        pipe = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", model)])
        s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["r2"], n_jobs=-1)
        comparison.append({"modele": name, "r2_cv": round(float(s["test_r2"].mean()), 4),
                           "r2_cv_std": round(float(s["test_r2"].std()), 4)})
        print(f"{name:22s} R² CV = {s['test_r2'].mean():.4f}")

    params = BEST_PARAMS
    if grid:
        search = GridSearchCV(
            Pipeline([("preprocessor", make_preprocessor(X_train)),
                      ("model", XGBRegressor(subsample=0.8, colsample_bytree=0.8,
                                             random_state=RANDOM_STATE, n_jobs=-1))]),
            {"model__n_estimators": [300, 500, 800], "model__max_depth": [3, 4, 6],
             "model__learning_rate": [0.03, 0.05, 0.1]},
            cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        search.fit(X_train, y_train)
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        print("Meilleurs hyperparamètres :", params)

    pipe = Pipeline([("preprocessor", make_preprocessor(X_train)),
                     ("model", XGBRegressor(subsample=0.8, colsample_bytree=0.8,
                                            random_state=RANDOM_STATE, n_jobs=-1, **params))])
    pipe.fit(X_train, y_train)
    pred = np.expm1(pipe.predict(X_test))
    metrics = {
        "rmse_dollars": round(float(np.sqrt(mean_squared_error(p_test, pred))), 0),
        "mae_dollars": round(float(mean_absolute_error(p_test, pred)), 0),
        "r2": round(float(r2_score(p_test, pred)), 4),
        "mape_pct": round(float(np.mean(np.abs((p_test.values - pred) / p_test.values)) * 100), 2),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "hyperparametres": params, "comparaison_cv": comparison,
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "comparaison_cv"}, ensure_ascii=False))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "columns": X.columns.tolist()}, MODELS_DIR / "prix.joblib")
    pd.DataFrame({"prix_reel": p_test.values, "prix_predit": pred}).to_csv(
        MODELS_DIR / "prix_test_predictions.csv", index=False)
    (MODELS_DIR / "prix_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    main(grid="--grid" in sys.argv)
