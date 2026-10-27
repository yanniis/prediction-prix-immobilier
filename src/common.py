"""Chargement et préparation des données Ames Housing, communs aux 4 parties du projet.

Le nettoyage reprend celui des notebooks d'origine :
- noms de colonnes avec des underscores,
- suppression des identifiants techniques,
- suppression des 5 ventes atypiques (surface > 4000 pieds² vendues < 300 000 $),
  recommandée par l'auteur du jeu de données,
- valeurs manquantes qui signifient « pas d'équipement » remplacées par "None" ou 0,
- variables dérivées (surface totale, âge, salles de bain…).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "AmesHousing.csv"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42

NONE_COLS = ["Alley", "Bsmt_Qual", "Bsmt_Cond", "Bsmt_Exposure", "BsmtFin_Type_1", "BsmtFin_Type_2",
             "Fireplace_Qu", "Garage_Type", "Garage_Finish", "Garage_Qual", "Garage_Cond",
             "Pool_QC", "Fence", "Misc_Feature", "Mas_Vnr_Type"]
ZERO_COLS = ["Mas_Vnr_Area", "BsmtFin_SF_1", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF",
             "Bsmt_Full_Bath", "Bsmt_Half_Bath", "Garage_Cars", "Garage_Area"]
DERIVED = ["Total_SF", "House_Age", "Years_Since_Remod", "Total_Bathrooms",
           "Has_Pool", "Has_Garage", "Has_Fireplace"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().replace(" ", "_").replace("/", "_") for c in df.columns]
    df = df.drop(columns=[c for c in ["Unnamed:_0", "Order", "PID"] if c in df.columns])
    # ventes atypiques recommandées à l'exclusion par l'auteur du jeu de données
    df = df[~((df["Gr_Liv_Area"] > 4000) & (df["SalePrice"] < 300000))].reset_index(drop=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NONE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("None")
    for c in ZERO_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    if "Garage_Yr_Blt" in df.columns:
        df["Garage_Yr_Blt"] = df["Garage_Yr_Blt"].fillna(df["Year_Built"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Total_SF"] = df["Total_Bsmt_SF"] + df["1st_Flr_SF"] + df["2nd_Flr_SF"]
    df["House_Age"] = (df["Yr_Sold"] - df["Year_Built"]).clip(lower=0)
    df["Years_Since_Remod"] = (df["Yr_Sold"] - df["Year_Remod_Add"]).clip(lower=0)
    df["Total_Bathrooms"] = (df["Full_Bath"] + 0.5 * df["Half_Bath"]
                             + df["Bsmt_Full_Bath"] + 0.5 * df["Bsmt_Half_Bath"])
    df["Has_Pool"] = (df["Pool_Area"] > 0).astype(int)
    df["Has_Garage"] = (df["Garage_Area"] > 0).astype(int)
    df["Has_Fireplace"] = (df["Fireplaces"] > 0).astype(int)
    return df


def load_dataset() -> pd.DataFrame:
    """Jeu de données nettoyé et enrichi, prêt pour les 4 parties."""
    return add_features(clean(load_raw()))


def typical_house(df: pd.DataFrame) -> dict:
    """Une maison « typique » : médiane des colonnes numériques, valeur la plus fréquente des autres.

    Sert de base aux caractéristiques que l'utilisateur ne renseigne pas dans l'interface.
    """
    base = {}
    for c in df.columns:
        if c == "SalePrice":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            base[c] = float(df[c].median())
        else:
            base[c] = df[c].mode(dropna=True).iloc[0]
    return base


def to_frame(house: dict) -> pd.DataFrame:
    """Transforme une maison décrite par un dictionnaire en tableau, avec ses variables dérivées."""
    return add_features(pd.DataFrame([house]))
