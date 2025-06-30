from flask import Flask, render_template, request
import pandas as pd
import json
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ---- 1. Charger les données au lancement ----
with open('COVID_19BXL.json', 'r', encoding='utf-8') as f:
    dico = json.load(f)

records = []
for date, communes in dico.items():
    for commune, valeur in communes.items():
        records.append({'date': date, 'commune': commune, 'valeur': int(valeur)})

df = pd.DataFrame(records)
df['date'] = pd.to_datetime(df['date'])

def moyenne_glissante(df, n=7):
    df['valeur'] = pd.to_numeric(df['valeur'], errors='coerce').fillna(0)
    df = df.sort_values(by=['date', 'commune'])
    df['moyenne_glissante'] = df.groupby('commune')['valeur'].transform(
        lambda x: x.rolling(window=n, min_periods=1).mean())
    return df

donnee_lissee = moyenne_glissante(df)

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

# ---- 2. Routes Flask ----
@app.route('/', methods=['GET', 'POST'])
def index():
    communes = sorted(df['commune'].unique())
    result = None
    plot_url = None
    table_html = None

    if request.method == 'POST':
        commune = request.form['commune']
        date_debut = request.form['date_debut']
        date_fin = request.form['date_fin']

        # Données filtrées pour la commune et période
        mask = (
            (donnee_lissee['commune'] == commune) &
            (donnee_lissee['date'] >= pd.to_datetime(date_debut)) &
            (donnee_lissee['date'] <= pd.to_datetime(date_fin))
        )
        donnee_commune = donnee_lissee[mask].copy()
        prev = prevision(date_debut, date_fin, donnee_lissee, commune)

        # Préparer le graphe (réel + moyenne glissante + prévision)
        plt.figure(figsize=(10,5))
        plt.plot(donnee_commune['date'], donnee_commune['valeur'], label='Valeur réelle', marker='o')
        plt.plot(donnee_commune['date'], donnee_commune['moyenne_glissante'], label='Moyenne glissante', linestyle='--')
        plt.plot(prev['date'], prev['valeur_prevue'], label='Prévision', color='red')
        plt.xlabel('Date')
        plt.ylabel('Valeur')
        plt.title(f"Commune: {commune} — du {date_debut} au {date_fin}")
        plt.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')

        # Table HTML
        donnee_commune['date'] = donnee_commune['date'].dt.strftime('%Y-%m-%d')
        prev['date'] = prev['date'].dt.strftime('%Y-%m-%d')
        table_html = donnee_commune[['date','valeur','moyenne_glissante']].to_html(index=False)
        prev_html = prev[['date','valeur_prevue']].to_html(index=False)

        result = True
    else:
        commune = communes[0]
        date_debut = str(df['date'].min().date())
        date_fin = str(df['date'].max().date())

    return render_template(
        'index.html',
        communes=communes,
        commune=commune,
        date_debut=date_debut,
        date_fin=date_fin,
        result=result,
        plot_url=plot_url,
        table_html=table_html,
        prev_html=prev_html if result else None
    )

if __name__ == '__main__':
    app.run(debug=True)
