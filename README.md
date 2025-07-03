# Chaîne de Markov sur les paliers COVID par commune (visualisation stylée)

Cette application Streamlit permet d’analyser et de prédire les niveaux d’incidence COVID-19 (“paliers” faible, moyen, élevé) par commune (Région Bruxelles), en utilisant une chaîne de Markov et des visualisations interactives Plotly.

## Installation & Lancement

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-utilisateur/markov-covid-communes.git
cd markov-covid-communes
pip install -r requirements.txt
pip install streamlit pandas numpy plotly
streamlit run app.py
