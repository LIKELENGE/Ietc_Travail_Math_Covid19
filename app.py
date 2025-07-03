import json               # Pour lire les fichiers JSON
import pandas as pd       # Pour manipuler des tableaux de données
import numpy as np        # Pour les calculs numériques, tableaux, etc.
import streamlit as st    # Pour créer l’application web interactive
import plotly.graph_objs as go  # Pour les graphiques stylés et interactifs

st.title("Chaîne de Markov sur les paliers COVID par commune (visualisation stylée)")
# Affiche le titre principal de l'application dans la page Streamlit

# ============ 1. Chargement des données =============
with open("COVID_19BXL.json", "r") as json_file:
    data = json.load(json_file)
    # Ouvre le fichier JSON et charge son contenu dans une variable Python

df = pd.DataFrame.from_dict(data, orient='index')
# Transforme le dictionnaire en DataFrame Pandas, avec les dates comme index

df = df.reset_index().rename(columns={'index': 'date'})
# Remet l'index en colonne et renomme-la "date"

df = df.fillna(0)
# Remplace les valeurs manquantes (NaN) par 0

for col in df.columns:
    if col != 'date':
        df[col] = df[col].astype(int)
        # Convertit toutes les colonnes (sauf la date) en entiers

df['date'] = pd.to_datetime(df['date'])
# Convertit la colonne "date" au format datetime

# ============ 2. Sélection commune, lissage et paliers =============
communes = [col for col in df.columns if col not in ['date']]
# Liste toutes les colonnes sauf "date" pour obtenir les noms des communes

commune = st.sidebar.selectbox("Commune à afficher", options=["Toutes"] + communes, index=0)
# Crée un menu déroulant dans la barre latérale pour choisir la commune à visualiser

def get_palier(nb_cas):
    if nb_cas < 50:
        return 'Faible'
    elif nb_cas < 200:
        return 'Moyen'
    else:
        return 'Élevé'
# Fonction qui attribue un "palier" (faible/moyen/élevé) selon le nombre de cas

if commune == "Toutes":
    df['total_cas'] = df.drop('date', axis=1).sum(axis=1)
    # Calcule le total des cas sur toutes les communes pour chaque date
    df['val_cas'] = df['total_cas']
    # Stocke ce total dans une colonne commune (pour la suite)
else:
    df['val_cas'] = df[commune]
    # Sinon, ne prend que les valeurs de la commune sélectionnée

df['cas_lisse'] = df['val_cas'].rolling(window=7, min_periods=1).mean()
# Lisse la série des cas sur 7 jours (moyenne mobile), pour réduire le bruit

df['palier'] = df['cas_lisse'].apply(get_palier)
# Attribue à chaque jour un palier (faible, moyen, élevé) selon les cas lissés

paliers = ['Faible', 'Moyen', 'Élevé']
# Liste ordonnée des différents paliers

# ============ 3. Sélection de l'année pour calculer P ============
annees = sorted(df['date'].dt.year.unique())
# Récupère la liste des années présentes dans les données

annee_P = st.sidebar.selectbox("Année pour calcul de P", options=[str(a) for a in annees], index=0)
# Menu déroulant pour choisir l'année sur laquelle on veut calculer la matrice de transition

df_hist = df[df['date'].dt.year == int(annee_P)].copy()
# Sélectionne uniquement les données correspondant à l'année choisie

# ============ 3 bis. Matrice initiale par année sélectionnée ============
init_count = df_hist['palier'].value_counts().reindex(paliers, fill_value=0)
# Compte le nombre de jours dans chaque palier, pour l'année sélectionnée

init_proba = (init_count / init_count.sum()).round(3)
# Calcule la proportion (probabilité) initiale de chaque palier

st.subheader(f"Distribution initiale des paliers sur l'année {annee_P}")
# Affiche un sous-titre dans l’application

st.dataframe(pd.DataFrame({
    "Nombre de jours": init_count,
    "Probabilité initiale": init_proba
}).T)
# Affiche un tableau de la distribution initiale des paliers

# Construction de la matrice de transition P sur l'année sélectionnée
serie_paliers = df_hist['palier']
# Série de paliers (faible/moyen/élevé) sur l'année sélectionnée

mat_count = pd.DataFrame(
    np.zeros((len(paliers), len(paliers)), dtype=int),
    index=paliers, columns=paliers
)
# Crée une matrice carrée de zéros, qui va compter les transitions entre paliers

for p1, p2 in zip(serie_paliers[:-1], serie_paliers[1:]):
    mat_count.loc[p1, p2] += 1
# Parcourt les paliers deux à deux et incrémente le nombre de transitions observées

P = mat_count.div(mat_count.sum(axis=1), axis=0).fillna(0)
# Transforme les comptes en probabilités (chaque ligne = 1), pour obtenir la matrice de transition P

st.subheader(f"Matrice de transition (P) calculée sur l'année {annee_P}")
# Affiche le sous-titre

st.dataframe(P)
# Affiche la matrice de transition sous forme de tableau

# ============ 4. Sélection de la période de prédiction ============
st.subheader("Sélection de la période à prédire")
# Affiche un sous-titre pour cette étape

date_min = df['date'].min().date()
date_max = df['date'].max().date()
# Récupère les dates extrêmes du jeu de données

periode = st.date_input(
    "Période de prédiction",
    value=(date_min, date_min if date_min == date_max else date_min + pd.Timedelta(days=7)),
    min_value=date_min,
    max_value=date_max
)
# Widget pour choisir la période à prédire (par défaut : une semaine après la première date)

if not isinstance(periode, tuple):
    periode = (periode, periode)
# Si l’utilisateur ne sélectionne qu’une date, on transforme en tuple (début, fin)

date_debut_pred = pd.to_datetime(periode[0])
date_fin_pred = pd.to_datetime(periode[1])
# Transforme les dates choisies en objet datetime (pour filtrer le dataframe)

# ============ 5. Générer la séquence prédite =============
df_pred = df[(df['date'] >= date_debut_pred) & (df['date'] <= date_fin_pred)].copy()
# Sélectionne la portion du dataframe correspondant à la période choisie

df_pred = df_pred.drop_duplicates(subset='date')
# Retire les éventuels doublons de dates

if df_pred.empty:
    st.warning("Aucune donnée pour la période sélectionnée.")
    st.stop()
# Si aucune donnée sur cette période, affiche un message d’avertissement et arrête l’exécution

dates_pred = df_pred['date'].tolist()
# Liste des dates à prédire

paliers_reels = df_pred['palier'].tolist()
# Liste réelle des paliers sur cette période (pour comparer à la prédiction)

etat_courant = paliers_reels[0]
# On commence la prédiction au premier palier réel de la période

predits = [etat_courant]
# On initialise la séquence prédite avec ce premier état

for i in range(1, len(paliers_reels)):
    proba = P.loc[etat_courant].values
    # Récupère la ligne correspondante dans la matrice de transition (proba de changer de palier)
    if np.sum(proba) == 0:
        etat_suivant = np.random.choice(paliers)
        # Si la ligne ne contient que des zéros (pas de transitions observées), choix aléatoire
    else:
        etat_suivant = np.random.choice(paliers, p=proba)
        # Sinon, tire au sort le prochain état selon la distribution de proba donnée par la matrice
    predits.append(etat_suivant)
    etat_courant = etat_suivant
    # Ajoute le nouvel état à la séquence, et continue avec ce nouvel état

assert len(dates_pred) == len(paliers_reels) == len(predits), "Longueurs différentes ! Vérifie tes index."
# Vérifie que toutes les listes ont la même taille (sécurité)

# ============ 6. Comparaison ============
nb_bonnes = sum([a == b for a, b in zip(paliers_reels, predits)])
# Compte le nombre de fois où la prédiction est correcte

taux_bonne_pred = nb_bonnes / len(predits)
# Calcule le taux de bonne prédiction

st.subheader("Comparaison Réel / Prédiction Markov (sur paliers lissés)")
# Affiche un sous-titre

col1, col2 = st.columns(2)
# Crée deux colonnes pour afficher les séquences réel/prédit côte à côte

col1.write("Séquence réelle :")
col1.write(paliers_reels)
# Affiche la séquence réelle des paliers

col2.write("Séquence prédite :")
col2.write(predits)
# Affiche la séquence prédite par le modèle de Markov

st.write(f"**Taux de bonne prédiction :** {taux_bonne_pred:.2%}")
# Affiche le pourcentage de réussite du modèle

# ============ 7. Affichage graphique stylé (2 courbes seulement) ============
palier_map = {'Faible': 0, 'Moyen': 1, 'Élevé': 2}
# Dictionnaire pour associer chaque palier à une valeur numérique

palier_labels = ['Faible', 'Moyen', 'Élevé']
# Liste ordonnée pour l’affichage

predits_num = [palier_map[p] for p in predits]
# Transforme la séquence prédite (texte) en valeurs numériques pour le graphe

fig = go.Figure()
# Initialise un objet Figure pour le graphique Plotly

# Courbe des cas lissés (ligne continue, axe gauche)
fig.add_trace(go.Scatter(
    x=df_pred['date'],
    y=df_pred['cas_lisse'],
    mode='lines+markers',
    name='Cas lissés (7j)',
    line=dict(color='#3498db', width=3, dash='solid'),
    marker=dict(size=7, color='#2980b9'),
    yaxis='y1'
))
# Ajoute la courbe des cas lissés : dates en x, cas lissés en y, ligne bleue, points, axe gauche

# Courbe des paliers prédits par Markov (ligne escalier/points, axe droite)
fig.add_trace(go.Scatter(
    x=dates_pred,
    y=predits_num,
    mode='lines+markers',
    name='Palier Markov (prédit)',
    line=dict(color='#e67e22', width=3, shape='hv'),
    marker=dict(size=9, color='#e67e22', symbol='diamond'),
    yaxis='y2'
))
# Ajoute la courbe des paliers prédits : dates en x, valeurs numériques en y, ligne orange, points diamants, axe droit

fig.update_layout(
    template='plotly_dark',
    plot_bgcolor='rgba(40, 40, 70, 0.9)',
    title=dict(
        text="Cas lissés (7j) et Paliers prédits (Markov)",
        x=0.5, font=dict(size=22)
    ),
    xaxis=dict(title="Date", tickangle=-30),
    yaxis=dict(
        title="Cas lissés",
        side="left",
        showgrid=True,
        zeroline=False,
    ),
    yaxis2=dict(
        title="Palier prédit",
        side="right",
        overlaying='y',
        tickmode='array',
        tickvals=[0,1,2],
        ticktext=palier_labels,
        range=[-0.3,2.3],
        showgrid=False
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=50, t=80, b=60),
    hovermode="x unified",
)
# Personnalise l’apparence du graphique : fond, titre, axes, échelles, légende, marges, survol…

st.plotly_chart(fig, use_container_width=True)
# Affiche le graphique interactif dans la page Streamlit (en large)

