"""Interface Streamlit du projet immobilier Ames (projet de groupe, Mastère Big Data & IA – IPSSI).

Lancer :  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from common import MODELS_DIR, load_dataset  # noqa: E402
from estimation import (CUISINE, M2, NOMS, QUARTIERS, charger_prix, construire_maison,  # noqa: E402
                        estimer)
from train_recommandation import recommander  # noqa: E402

VIOLET, MAGENTA, CYAN, ROUGE = "#7f1fcf", "#b51aa6", "#0b8f9a", "#d64545"

st.set_page_config(page_title="Immobilier Ames · IA", page_icon="🏠", layout="wide")


# ---------------------------------------------------------------- chargement (mis en cache)
@st.cache_resource
def charger():
    return {
        **charger_prix(), "df": load_dataset(),
        "clust": joblib.load(MODELS_DIR / "clustering.joblib"),
        "type": joblib.load(MODELS_DIR / "type_logement.joblib"),
        "points": pd.read_csv(MODELS_DIR / "clustering_points.csv"),
        "profils": pd.read_csv(MODELS_DIR / "clustering_profils.csv"),
        "catalogue": pd.read_csv(MODELS_DIR / "recommandation_catalogue.csv"),
        "test": pd.read_csv(MODELS_DIR / "prix_test_predictions.csv"),
        "m": {k: json.loads((MODELS_DIR / f"{k}_metrics.json").read_text())
              for k in ["prix", "clustering", "type_logement", "recommandation"]},
    }


R = charger()
df, mp = R["df"], R["m"]["prix"]


def dollars(x: float) -> str:
    return f"{x:,.0f} $".replace(",", " ")


def m2(pieds2: float) -> str:
    return f"{pieds2 / M2:,.0f} m²".replace(",", " ")


# ---------------------------------------------------------------- description de la maison
with st.sidebar:
    st.header("🏠 Décrire la maison")
    st.caption("Les caractéristiques non demandées prennent la valeur la plus courante du jeu de données.")
    codes = sorted(QUARTIERS, key=lambda c: QUARTIERS[c])
    quartier = st.selectbox("Quartier", codes, index=codes.index("CollgCr"),
                            format_func=lambda c: QUARTIERS[c])
    niveaux = st.radio("Niveaux", ["Plain-pied", "Deux niveaux"], horizontal=True, index=1)
    surf = st.number_input("Surface habitable (m²)", 40, 450, 160, step=5)
    sous_sol = st.number_input("Surface du sous-sol (m²)", 0, 300, 80, step=5)
    terrain = st.number_input("Surface du terrain (m²)", 100, 10000, 900, step=50)
    qual = st.slider("Qualité globale des matériaux (1 à 10)", 1, 10, 7)
    etat = st.slider("État général (1 à 10)", 1, 10, 5)
    c1, c2 = st.columns(2)
    annee = c1.number_input("Construite en", 1872, 2010, 2000)
    renov = c2.number_input("Rénovée en", int(annee), 2010, max(int(annee), 2000))
    c1, c2 = st.columns(2)
    chambres = c1.number_input("Chambres", 0, 8, 3)
    sdb = c2.number_input("Salles de bain", 0, 4, 2)
    c1, c2 = st.columns(2)
    garage = c1.number_input("Places de garage", 0, 4, 2)
    chem = c2.number_input("Cheminées", 0, 3, 1)
    cuisine = st.select_slider("Qualité de la cuisine", list(CUISINE)[::-1], value="Bonne")
    clim = st.toggle("Climatisation centrale", value=True)

maison = construire_maison(
    R["base"], quartier=quartier, plain_pied=(niveaux == "Plain-pied"), surf=surf, sous_sol=sous_sol,
    terrain=terrain, qual=qual, etat=etat, annee=annee, renov=renov, chambres=chambres, sdb=sdb,
    garage=garage, chem=chem, cuisine=CUISINE[cuisine], clim=clim)

# ---------------------------------------------------------------- calculs
X, prix, contrib = estimer(R, maison)

clust = R["clust"]
cluster = int(clust["kmeans"].predict(clust["scaler"].transform(X[clust["features"]]))[0])
segment = clust["label_map"][cluster]
pc = clust["pca"].transform(clust["scaler"].transform(X[clust["features"]]))[0]

typ = R["type"]
Xc = X.assign(SalePrice=prix)[typ["columns"]]
probas = pd.Series(typ["pipeline"].predict_proba(Xc)[0], index=typ["label_encoder"].classes_).sort_values(ascending=False)

# ---------------------------------------------------------------- en-tête
st.title("Immobilier Ames · estimation, profils et bonnes affaires")
st.markdown(
    "Projet de groupe du **Mastère Big Data & IA (IPSSI)** : quatre modèles d'IA qui travaillent sur les "
    "mêmes 2 927 ventes de maisons. Décrivez une maison dans le panneau de gauche (bouton » sur "
    "mobile) : chaque onglet la "
    "regarde sous un angle différent.")
st.caption("Données : ventes réelles à Ames (Iowa, États-Unis) entre 2006 et 2010 · prix en dollars.")

t1, t2, t3, t4, t5 = st.tabs(["💰 Estimer le prix", "🧩 Profil du logement", "🏷️ Type de logement",
                              "🔎 Trouver une bonne affaire", "ℹ️ À propos"])

# ---------------------------------------------------------------- 1. prix
with t1:
    st.caption("Partie réalisée par **Yannis Bouttier**")
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        st.metric("Prix estimé", dollars(prix))
        st.markdown(f"En moyenne, le modèle se trompe de **{dollars(mp['mae_dollars'])}** "
                    f"(**{mp['mape_pct']:.0f} %**) sur des maisons qu'il n'a jamais vues.")
        st.subheader("Pourquoi ce prix ?")
        for var, v in contrib.head(3).items():
            pct = (np.exp(v) - 1) * 100
            sens = "fait monter" if v > 0 else "fait baisser"
            st.markdown(f"- **{NOMS.get(var, var)}** {sens} le prix d'environ **{abs(pct):.0f} %**")
        st.caption("Calculé avec SHAP : la part de chaque caractéristique dans l'écart entre ce prix "
                   "et le prix moyen des maisons du jeu de données.")
    with c2:
        top = contrib.head(8)[::-1]
        pct = (np.exp(top.values) - 1) * 100
        fig = go.Figure(go.Bar(
            x=pct, y=[NOMS.get(v, v) for v in top.index], orientation="h",
            marker_color=[VIOLET if p > 0 else ROUGE for p in pct],
            text=[f"{p:+.0f} %" for p in pct], textposition="outside", cliponaxis=False))
        fig.update_layout(title="Effet des 8 caractéristiques qui pèsent le plus", height=380,
                          margin=dict(l=10, r=40, t=50, b=10), xaxis_title="Effet sur le prix (%)",
                          xaxis_zeroline=True)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Comment le modèle a été construit"):
        st.markdown(
            "Cinq modèles ont été comparés en validation croisée (on teste chaque modèle 5 fois sur des "
            "morceaux différents des données), du plus simple au plus avancé. XGBoost a été retenu puis "
            f"évalué sur **{mp['n_test']} maisons mises de côté** dès le départ.")
        comp = pd.DataFrame(mp["comparaison_cv"]).rename(columns={
            "modele": "Modèle", "r2_cv": "R² moyen (validation croisée)", "r2_cv_std": "Écart-type"})
        st.dataframe(comp.sort_values("R² moyen (validation croisée)", ascending=False),
                     hide_index=True, use_container_width=True)
        cc = st.columns(4)
        cc[0].metric("R² (test)", f"{mp['r2']:.2f}")
        cc[1].metric("Erreur moyenne", dollars(mp["mae_dollars"]))
        cc[2].metric("Erreur relative", f"{mp['mape_pct']:.1f} %")
        cc[3].metric("RMSE", dollars(mp["rmse_dollars"]))
        t = R["test"]
        fig = px.scatter(t, x="prix_reel", y="prix_predit", opacity=0.5,
                         labels={"prix_reel": "Prix réel ($)", "prix_predit": "Prix estimé ($)"},
                         color_discrete_sequence=[VIOLET])
        lim = [t.min().min(), t.max().max()]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color=ROUGE),
                                 name="Estimation parfaite"))
        fig.update_layout(title="Prix estimé et prix réel sur les maisons de test", height=420)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 2. clustering
with t2:
    st.caption("Partie réalisée par **Fabrice Robert**")
    st.markdown(f"Cette maison appartient au profil **{segment}**.")
    st.markdown(
        "Un algorithme de clustering (KMeans) a regroupé les 2 927 maisons en **9 profils**, uniquement à "
        "partir de leurs caractéristiques physiques (taille, âge, qualité, confort), **sans jamais voir le "
        "prix**. Le prix médian de chaque profil sert ensuite à vérifier que les groupes ont du sens.")
    pts = R["points"]
    fig = px.scatter(pts, x="pc1", y="pc2", color="type_logement", opacity=0.45,
                     labels={"pc1": "Axe 1 : plus grand, de meilleure qualité et plus récent →",
                             "pc2": "Axe 2 : plus de pièces et plus ancien →",
                             "type_logement": "Profil"},
                     color_discrete_sequence=px.colors.qualitative.Bold)
    fig.add_trace(go.Scatter(x=[pc[0]], y=[pc[1]], mode="markers", name="Votre maison",
                             marker=dict(symbol="star", size=26, color="#111", line=dict(color="white", width=2))))
    fig.add_annotation(x=pc[0], y=pc[1], text="<b>Votre maison</b>", showarrow=True, arrowhead=2,
                       ax=40, ay=-50, bgcolor="white", bordercolor="#111", borderpad=4)
    fig.update_layout(height=520, legend=dict(orientation="h", y=-0.2),
                      title="Les 2 927 maisons projetées sur 2 axes (ACP)")
    st.plotly_chart(fig, use_container_width=True)
    prof = R["profils"].copy()
    prof["Surface totale"] = (prof["surface_totale_pieds2"] / M2).round(0).astype(int).astype(str) + " m²"
    prof = prof.rename(columns={"type_logement": "Profil", "nb_logements": "Maisons",
                                "age_moyen": "Âge moyen (ans)", "qualite_moyenne": "Qualité moyenne (/10)",
                                "prix_median": "Prix médian ($)"})
    prof = prof.sort_values("Prix médian ($)", ascending=False)
    prof["Prix médian ($)"] = prof["Prix médian ($)"].map(dollars)
    st.dataframe(prof[["Profil", "Maisons", "Surface totale", "Âge moyen (ans)", "Qualité moyenne (/10)",
                       "Prix médian ($)"]],
                 hide_index=True, use_container_width=True)
    st.caption(f"9 groupes retenus pour obtenir des profils exploitables (le découpage mathématiquement "
               f"optimal, à 2 groupes, était trop grossier). Score de silhouette : "
               f"{R['m']['clustering']['silhouette_k9']:.2f}. Deux groupes portent le même nom : le nommage "
               "automatique les classe dans les mêmes tranches de taille, d'âge et de qualité.")

# ---------------------------------------------------------------- 3. classification
with t3:
    st.caption("Partie réalisée par **Duncan Costes**")
    mt = R["m"]["type_logement"]
    st.markdown(f"Le modèle classe cette maison dans la catégorie **{probas.index[0]}**, "
                f"avec une confiance de **{probas.iloc[0] * 100:.0f} %**.")
    st.markdown(
        "Ici, on apprend à **reconnaître le profil d'une maison à partir de sa fiche complète** (y compris "
        "son prix, ici le prix estimé dans le premier onglet), sans refaire le clustering. Trois modèles "
        f"ont été comparés ; la **{mt['modele_retenu'].lower()}** est retenue, avec "
        f"**{mt['accuracy_test'] * 100:.1f} %** de bonnes réponses sur des maisons de test.")
    top3 = probas.head(3)[::-1]
    fig = go.Figure(go.Bar(x=top3.values * 100, y=top3.index, orientation="h", marker_color=CYAN,
                           text=[f"{v * 100:.0f} %" for v in top3.values], textposition="outside",
                           cliponaxis=False))
    fig.update_layout(title="Les 3 catégories les plus probables", xaxis_title="Probabilité (%)",
                      xaxis_range=[0, 110], height=300, margin=dict(l=10, r=30, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    if probas.index[0] == segment:
        st.success("Le classifieur et le clustering sont d'accord sur le profil de cette maison.")
    else:
        st.info(f"Le clustering place cette maison dans « {segment} » : elle est proche de la frontière "
                "entre deux profils.")
    st.caption("Précaution prise : le numéro de groupe du clustering est retiré des variables, sinon la "
               "réponse serait donnée d'avance. La catégorie à 4 maisons seulement a été écartée.")

# ---------------------------------------------------------------- 4. recommandation
with t4:
    st.caption("Partie réalisée par **Abdellatif Essid**")
    st.markdown(
        "Un catalogue de **586 maisons en vente** est comparé à l'estimation d'un modèle d'IA. Quand le "
        "prix demandé est **inférieur à l'estimation**, la maison est peut-être une bonne affaire. Le "
        "**Deal Index** mesure de combien l'estimation dépasse le prix demandé : +20 % veut dire que "
        "l'IA estime la maison 20 % plus cher que son prix.")
    c = st.columns(4)
    budget = c[0].number_input("Budget maximum ($)", 50000, 800000, 260000, step=10000)
    mch = c[1].number_input("Chambres minimum", 0, 6, 3)
    mga = c[2].number_input("Places de garage minimum", 0, 4, 2)
    msb = c[3].number_input("Salles de bain minimum", 0, 4, 2)
    normales = st.toggle("Ignorer les ventes inhabituelles (entre proches, saisies…)", value=True,
                         help="Leur prix ne reflète pas le marché et donne de fausses bonnes affaires.")
    recos = recommander(R["catalogue"], budget, mch, mga, msb, ventes_normales=normales)
    if recos.empty:
        st.warning("Aucune maison ne correspond à ces critères. Essayez d'élargir la recherche.")
    else:
        cols = st.columns(len(recos))
        for col, (_, r) in zip(cols, recos.iterrows()):
            with col.container(border=True):
                st.markdown(f"**{QUARTIERS.get(r['Neighborhood'], r['Neighborhood'])}**")
                st.metric("Prix demandé", dollars(r["Prix"]), f"Deal Index {r['DealIndex']:+.0f} %")
                st.markdown(f"Estimation IA : **{dollars(r['Estimation'])}**  \n"
                            f"{int(r['Bedroom_AbvGr'])} chambres · {int(r['Full_Bath'])} salles de bain · "
                            f"{m2(r['Gr_Liv_Area'])} habitables  \n"
                            f"Qualité {int(r['Overall_Qual'])}/10 · construite en {int(r['Year_Built'])}")
        fig = go.Figure()
        noms = [f"{QUARTIERS.get(n, n)} (n°{i})" for n, i in zip(recos["Neighborhood"], recos["ID"])]
        fig.add_bar(name="Prix demandé", x=noms, y=recos["Prix"], marker_color=VIOLET)
        fig.add_bar(name="Estimation IA", x=noms, y=recos["Estimation"], marker_color=CYAN)
        fig.update_layout(barmode="group", height=340, yaxis_title="Dollars", title="Prix demandé et estimation")
        st.plotly_chart(fig, use_container_width=True)
    rm = R["m"]["recommandation"]
    st.caption(f"Classement : qualité × 10 + Deal Index, après les filtres. Modèle d'estimation : "
               f"{rm['modele_retenu']} sur 8 caractéristiques (erreur type de {dollars(rm['rmse_dollars'])}). "
               "Un Deal Index élevé invite à regarder le bien de plus près, ce n'est pas une garantie.")

# ---------------------------------------------------------------- 5. à propos
with t5:
    st.subheader("Le projet")
    st.markdown(
        "Projet de machine learning du Mastère Big Data & IA de l'IPSSI : concevoir un POC (une preuve de "
        "concept) pour une agence immobilière, à partir du jeu de données public *Ames Housing* "
        "(Dean De Cock, 2011).")
    st.dataframe(pd.DataFrame([
        ["Estimation du prix", "Yannis Bouttier", "XGBoost + explications SHAP",
         f"{mp['mape_pct']:.1f} % d'erreur moyenne, R² {mp['r2']:.2f}"],
        ["Profils de logements", "Fabrice Robert", "Clustering KMeans (9 profils)",
         "Profils cohérents avec le prix médian"],
        ["Type de logement", "Duncan Costes", "Régression logistique",
         f"{R['m']['type_logement']['accuracy_test'] * 100:.1f} % de bonnes réponses"],
        ["Recommandation", "Abdellatif Essid", "XGBoost + Deal Index", "Top 3 des bonnes affaires selon un budget"],
    ], columns=["Partie", "Réalisée par", "Méthode", "Résultat"]), hide_index=True, use_container_width=True)
    st.subheader("Limites")
    st.markdown(
        "- Les données couvrent une seule ville américaine entre 2006 et 2010 : les modèles ne sont pas "
        "utilisables tels quels sur un autre marché.\n"
        "- L'interface ne demande que les caractéristiques principales ; les autres prennent la valeur la "
        "plus courante du jeu de données.\n"
        "- Les surfaces sont saisies en m² et converties dans les unités d'origine (pieds²).")
