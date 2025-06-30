import pandas as pd
import json

with open('COVID_19BXL.json', 'r', encoding='utf-8') as f:
    dico = json.load(f)

records = []
for date, communes in dico.items():
    for commune, valeur in communes.items():
        records.append({'date': date, 'commune': commune, 'valeur': int(valeur)})

df = pd.DataFrame(records)
df['date'] = pd.to_datetime(df['date'])

def moyenne_glissante(df, n=7):
    """Calcule la moyenne glissante sur n jours pour chaque commune.
    Les erreurs de conversion sont remplacées par 0."""
    df['valeur'] = pd.to_numeric(df['valeur'], errors='coerce').fillna(0)
    df = df.sort_values(by=['date', 'commune'])
    df['moyenne_glissante'] = df.groupby('commune')['valeur'].transform(
        lambda x: x.rolling(window=n, min_periods=1).mean())
    return df

donnee_lissee = moyenne_glissante(df)

def matrice_transition(donnee_lissee):
    """Crée une matrice de transition globale entre les communes,
    en suivant simplement l'ordre chronologique des enregistrements."""
    # S'assurer que les données sont triées par date
    donnee_lissee = donnee_lissee.sort_values('date')
    communes = donnee_lissee['commune'].unique()
    transition_matrix = pd.DataFrame(0, index=communes, columns=communes)

    # Boucle sur toutes les lignes sauf la dernière
    for i in range(len(donnee_lissee) - 1):
        current_commune = donnee_lissee.iloc[i]['commune']
        next_commune = donnee_lissee.iloc[i + 1]['commune']
        transition_matrix.at[current_commune, next_commune] += 1

    return transition_matrix

import matplotlib.pyplot as plt
import pandas as pd

def prevision(date_debut, date_fin, donnee_lissee, commune):
    dates_prevision = pd.date_range(start=date_debut, end=date_fin, freq='D')
    donnee_commune = donnee_lissee[donnee_lissee['commune'] == commune]

    if donnee_commune.empty:
        raise ValueError(f"Aucune donnée pour la commune {commune}")

    last_moyenne = donnee_commune.sort_values('date').iloc[-1]['moyenne_glissante']

    df_prevision = pd.DataFrame({
        'date': dates_prevision,
        'commune': commune,
        'valeur_prevue': last_moyenne
    })

    return df_prevision

# Paramètres
commune = 'Ixelles'
date_debut = '2020-03-10'
date_fin = '2020-04-25'

# Prévision
df_prev = prevision(date_debut, date_fin, donnee_lissee, commune)

# Données réelles (uniquement la période de prévision)
mask = (donnee_lissee['commune'] == commune) & \
       (donnee_lissee['date'] >= pd.to_datetime(date_debut)) & \
       (donnee_lissee['date'] <= pd.to_datetime(date_fin))
donnee_commune = donnee_lissee[mask].copy()

plt.figure(figsize=(12, 6))

# Valeur réelle pendant la période de prévision
plt.plot(donnee_commune['date'], donnee_commune['valeur'], label='Valeur réelle', marker='o')

# Moyenne glissante pendant la période de prévision
plt.plot(donnee_commune['date'], donnee_commune['moyenne_glissante'], label='Moyenne glissante (observée)', linestyle='--')

# Prévision (période de prévision)
plt.plot(df_prev['date'], df_prev['valeur_prevue'], label='Prévision', linestyle='-', color='red')

plt.xlabel('Date')
plt.ylabel('Valeur')
plt.title(f"Prévision vs Réalité ({date_debut} à {date_fin}) pour {commune}")
plt.legend()
plt.tight_layout()
plt.show()
