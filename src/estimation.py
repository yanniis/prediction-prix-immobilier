"""Logique d'estimation du prix, partagée par l'interface Streamlit (app.py) et l'API (api.py).

Une seule source de vérité : on décrit une maison avec quelques caractéristiques (en m²), on la complète
avec la maison « typique » du jeu de données, puis le modèle XGBoost donne le prix et XGBoost lui-même
calcule la part de chaque caractéristique (SHAP) dans ce prix.
"""
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from common import MODELS_DIR, load_dataset, to_frame, typical_house

M2 = 10.7639  # pieds² par m²

QUARTIERS = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale", "BrkSide": "Brookside",
    "ClearCr": "Clear Creek", "CollgCr": "College Creek", "Crawfor": "Crawford", "Edwards": "Edwards",
    "Gilbert": "Gilbert", "Greens": "Greens", "GrnHill": "Green Hills", "IDOTRR": "Iowa DOT & Rail Road",
    "Landmrk": "Landmark", "MeadowV": "Meadow Village", "Mitchel": "Mitchell", "NAmes": "North Ames",
    "NPkVill": "Northpark Villa", "NWAmes": "Northwest Ames", "NoRidge": "Northridge",
    "NridgHt": "Northridge Heights", "OldTown": "Old Town", "SWISU": "Sud-ouest de l'université",
    "Sawyer": "Sawyer", "SawyerW": "Sawyer West", "Somerst": "Somerset", "StoneBr": "Stone Brook",
    "Timber": "Timberland", "Veenker": "Veenker",
}
NOMS = {  # noms lisibles des variables pour l'explication du prix
    "Total_SF": "Surface totale", "Gr_Liv_Area": "Surface habitable", "Overall_Qual": "Qualité globale",
    "Overall_Cond": "État général", "Neighborhood": "Quartier", "House_Age": "Âge de la maison",
    "Year_Built": "Année de construction", "Year_Remod_Add": "Année de rénovation",
    "Years_Since_Remod": "Ancienneté de la rénovation", "Total_Bsmt_SF": "Surface du sous-sol",
    "1st_Flr_SF": "Surface du rez-de-chaussée", "2nd_Flr_SF": "Surface de l'étage",
    "Lot_Area": "Surface du terrain", "Garage_Cars": "Places de garage", "Garage_Area": "Surface du garage",
    "Total_Bathrooms": "Salles de bain", "Full_Bath": "Salles de bain complètes", "Kitchen_Qual": "Qualité de la cuisine",
    "Fireplaces": "Cheminées", "Has_Fireplace": "Présence d'une cheminée", "Central_Air": "Climatisation",
    "Bedroom_AbvGr": "Chambres", "TotRms_AbvGrd": "Nombre de pièces", "MS_Zoning": "Zonage",
    "Bsmt_Qual": "Qualité du sous-sol", "Exter_Qual": "Qualité extérieure", "Functional": "Fonctionnalité",
    "Sale_Condition": "Condition de vente", "BsmtFin_SF_1": "Sous-sol aménagé", "Garage_Yr_Blt": "Année du garage",
    "Lot_Frontage": "Façade sur rue", "Bsmt_Exposure": "Exposition du sous-sol", "Heating_QC": "Qualité du chauffage",
    "Garage_Finish": "Finition du garage", "Fireplace_Qu": "Qualité de la cheminée", "Mas_Vnr_Area": "Parement en pierre",
    "Screen_Porch": "Véranda", "Wood_Deck_SF": "Terrasse en bois", "Open_Porch_SF": "Porche ouvert",
    "Condition_1": "Environnement proche", "Paved_Drive": "Allée pavée", "House_Style": "Style de maison",
    "Bldg_Type": "Type de bâtiment", "Foundation": "Fondations", "Has_Garage": "Présence d'un garage",
    "Exter_Cond": "État extérieur", "Sale_Type": "Type de vente", "Lot_Config": "Configuration du terrain",
    "Land_Contour": "Relief du terrain", "BsmtFin_Type_1": "Aménagement du sous-sol", "Roof_Style": "Style de toit",
    "Exterior_1st": "Revêtement extérieur", "Exterior_2nd": "Revêtement secondaire",
    "MS_SubClass": "Catégorie de logement", "Garage_Type": "Type de garage", "Kitchen_AbvGr": "Cuisines",
    "Half_Bath": "Demi-salles de bain", "Mo_Sold": "Mois de vente", "Yr_Sold": "Année de vente",
    "Bsmt_Unf_SF": "Sous-sol non aménagé", "Electrical": "Installation électrique", "Heating": "Chauffage",
    "Land_Slope": "Pente du terrain", "Street": "Accès routier", "Alley": "Ruelle", "Condition_2": "Environnement (2)",
    "Roof_Matl": "Matériau du toit", "Bsmt_Cond": "État du sous-sol", "Bsmt_Full_Bath": "Salles de bain au sous-sol",
    "Bsmt_Half_Bath": "Demi-salles de bain au sous-sol", "Pool_Area": "Surface de piscine", "Misc_Val": "Équipements divers",
    "Low_Qual_Fin_SF": "Surface de faible qualité", "Enclosed_Porch": "Porche fermé", "3Ssn_Porch": "Porche trois saisons",
    "Lot_Shape": "Forme du terrain", "Garage_Qual": "Qualité du garage", "Garage_Cond": "État du garage",
    "Pool_QC": "Qualité de la piscine", "Fence": "Clôture", "Misc_Feature": "Équipement divers",
    "Has_Pool": "Présence d'une piscine", "BsmtFin_SF_2": "Sous-sol aménagé (2)", "BsmtFin_Type_2": "Aménagement du sous-sol (2)",
    "Utilities": "Raccordements", "Mas_Vnr_Type": "Type de parement",
}
CUISINE = {"Excellente": "Ex", "Bonne": "Gd", "Moyenne": "TA", "Passable": "Fa"}


def charger_prix() -> dict:
    """Charge le modèle de prix et tout ce qu'il faut pour estimer une maison."""
    df = load_dataset()
    prix = joblib.load(MODELS_DIR / "prix.joblib")
    return {
        "base": typical_house(df),
        "prix": prix,
        "booster": prix["pipeline"].named_steps["model"].get_booster(),
        "metrics": json.loads((MODELS_DIR / "prix_metrics.json").read_text()),
    }


def construire_maison(base: dict, *, quartier: str, plain_pied: bool, surf: float, sous_sol: float,
                      terrain: float, qual: int, etat: int, annee: int, renov: int, chambres: int,
                      sdb: int, garage: int, chem: int, cuisine: str, clim: bool) -> dict:
    """Complète la maison typique avec les caractéristiques données (surfaces en m²)."""
    maison = dict(base)
    gr = surf * M2
    maison.update({
        "Neighborhood": quartier, "Gr_Liv_Area": gr, "Lot_Area": terrain * M2, "Overall_Qual": qual,
        "Overall_Cond": etat, "Year_Built": annee, "Year_Remod_Add": renov, "Bedroom_AbvGr": chambres,
        "Full_Bath": sdb, "Half_Bath": 0, "Garage_Cars": garage, "Garage_Area": garage * 260.0,
        "Fireplaces": chem, "Kitchen_Qual": cuisine, "Central_Air": "Y" if clim else "N",
        "Total_Bsmt_SF": sous_sol * M2, "BsmtFin_SF_1": sous_sol * M2 * 0.5, "Bsmt_Unf_SF": sous_sol * M2 * 0.5,
        "BsmtFin_SF_2": 0.0, "Bsmt_Full_Bath": 0, "Bsmt_Half_Bath": 0,
        "1st_Flr_SF": gr if plain_pied else gr / 2,
        "2nd_Flr_SF": 0.0 if plain_pied else gr / 2,
        "House_Style": "1Story" if plain_pied else "2Story",
        "TotRms_AbvGrd": max(chambres + 3, 3), "Garage_Yr_Blt": annee,
    })
    if sous_sol == 0:
        maison.update({c: "None" for c in ["Bsmt_Qual", "Bsmt_Cond", "Bsmt_Exposure", "BsmtFin_Type_1", "BsmtFin_Type_2"]})
    if garage == 0:
        maison.update({c: "None" for c in ["Garage_Type", "Garage_Finish", "Garage_Qual", "Garage_Cond"]})
        maison["Garage_Yr_Blt"] = annee
    if chem == 0:
        maison["Fireplace_Qu"] = "None"
    return maison


def estimer(R: dict, maison: dict) -> tuple[pd.DataFrame, float, pd.Series]:
    """Renvoie (tableau de la maison, prix en dollars, contributions SHAP par variable, triées)."""
    X = to_frame(maison)
    pipe = R["prix"]["pipeline"]
    Xp = X[R["prix"]["columns"]]
    prix = float(np.expm1(pipe.predict(Xp)[0]))

    pre = pipe.named_steps["preprocessor"]
    # contributions SHAP calculées par XGBoost lui-même, sur les données au même format qu'à
    # l'entraînement (matrice creuse) : leur somme redonne exactement la prédiction
    sv = R["booster"].predict(xgb.DMatrix(pre.transform(Xp)), pred_contribs=True)[0][:-1]
    cat_cols = [c for name, _, cols in pre.transformers_ if name == "cat" for c in cols]
    contrib = {}
    for fname, v in zip(pre.get_feature_names_out(), sv):
        kind, rest = fname.split("__", 1)
        var = rest if kind == "num" else max((c for c in cat_cols if rest.startswith(c + "_")), key=len)
        contrib[var] = contrib.get(var, 0.0) + float(v)
    contrib = pd.Series(contrib).sort_values(key=np.abs, ascending=False)
    return X, prix, contrib


def raisons(contrib: pd.Series, n: int = 3) -> list[dict]:
    """Les n caractéristiques qui pèsent le plus, avec leur effet en % sur le prix."""
    out = []
    for var, v in contrib.head(n).items():
        pct = (np.exp(v) - 1) * 100
        out.append({"facteur": NOMS.get(var, var), "variable": var,
                    "effet": "hausse" if v > 0 else "baisse", "effet_pct": round(float(pct), 1)})
    return out
