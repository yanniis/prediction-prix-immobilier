"""API d'estimation du prix d'une maison (partie de Yannis Bouttier).

Lancement :  uvicorn api:app --reload      puis ouvrir http://localhost:8000/docs

Une requête décrit la maison (surfaces en m²) ; la réponse donne le prix estimé en dollars, la marge
d'erreur habituelle du modèle et les 3 raisons principales de ce prix.
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from estimation import CUISINE, QUARTIERS, charger_prix, construire_maison, estimer, raisons  # noqa: E402

ETAT: dict = {}


@asynccontextmanager
async def cycle_de_vie(_: FastAPI):
    ETAT["R"] = charger_prix()  # le modèle est chargé une seule fois, au démarrage
    yield
    ETAT.clear()


app = FastAPI(
    title="Estimation du prix d'une maison (Ames, Iowa)",
    description="Modèle XGBoost entraîné sur 2 927 ventes réelles. Renvoie le prix estimé et les "
                "3 caractéristiques qui pèsent le plus dans ce prix.",
    version="1.0.0", lifespan=cycle_de_vie)


class Maison(BaseModel):
    quartier: str = Field("CollgCr", description="Code du quartier (voir GET /quartiers)", examples=["CollgCr"])
    niveaux: Literal["plain-pied", "deux niveaux"] = "deux niveaux"
    surface_m2: float = Field(160, ge=40, le=450, description="Surface habitable")
    sous_sol_m2: float = Field(80, ge=0, le=300, description="Surface du sous-sol (0 s'il n'y en a pas)")
    terrain_m2: float = Field(900, ge=100, le=10000, description="Surface du terrain")
    qualite: int = Field(7, ge=1, le=10, description="Qualité globale des matériaux et de la finition")
    etat: int = Field(5, ge=1, le=10, description="État général")
    annee_construction: int = Field(2000, ge=1872, le=2010)
    annee_renovation: int = Field(2000, ge=1872, le=2010)
    chambres: int = Field(3, ge=0, le=8)
    salles_de_bain: int = Field(2, ge=0, le=4)
    places_garage: int = Field(2, ge=0, le=4)
    cheminees: int = Field(1, ge=0, le=3)
    qualite_cuisine: Literal["Excellente", "Bonne", "Moyenne", "Passable"] = "Bonne"
    climatisation: bool = True


class Raison(BaseModel):
    facteur: str
    effet: Literal["hausse", "baisse"]
    effet_pct: float = Field(description="Effet de cette caractéristique sur le prix, en %")


class Estimation(BaseModel):
    prix_estime_usd: int
    erreur_moyenne_usd: int = Field(description="Erreur moyenne du modèle sur des maisons jamais vues")
    erreur_moyenne_pct: float
    fourchette_usd: tuple[int, int] = Field(description="Prix estimé ± l'erreur moyenne")
    trois_raisons: list[Raison]


@app.get("/", include_in_schema=False)
def accueil():
    return {"message": "API d'estimation immobilière. Documentation interactive : /docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/quartiers")
def quartiers():
    """Codes de quartier acceptés et leur nom lisible."""
    return QUARTIERS


@app.post("/predict", response_model=Estimation)
def predict(m: Maison):
    if m.quartier not in QUARTIERS:
        raise HTTPException(422, f"Quartier inconnu : {m.quartier!r}. Voir GET /quartiers.")
    if m.annee_renovation < m.annee_construction:
        raise HTTPException(422, "annee_renovation ne peut pas précéder annee_construction.")
    R = ETAT["R"]
    maison = construire_maison(
        R["base"], quartier=m.quartier, plain_pied=(m.niveaux == "plain-pied"), surf=m.surface_m2,
        sous_sol=m.sous_sol_m2, terrain=m.terrain_m2, qual=m.qualite, etat=m.etat,
        annee=m.annee_construction, renov=m.annee_renovation, chambres=m.chambres, sdb=m.salles_de_bain,
        garage=m.places_garage, chem=m.cheminees, cuisine=CUISINE[m.qualite_cuisine], clim=m.climatisation)
    _, prix, contrib = estimer(R, maison)
    mae = R["metrics"]["mae_dollars"]
    return Estimation(
        prix_estime_usd=round(prix), erreur_moyenne_usd=round(mae), erreur_moyenne_pct=round(R["metrics"]["mape_pct"], 1),
        fourchette_usd=(round(prix - mae), round(prix + mae)), trois_raisons=raisons(contrib, 3))
