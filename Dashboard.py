#-----------------------#
# IMPORT DES LIBRAIRIES #
#-----------------------#

import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)
import shap
import requests as re
import numpy as np


#---------------------#
# VARIABLES STATIQUES #
#---------------------#

API_PRED = "https://api-creditscore.herokuapp.com/predict/"
API_SHAP = "https://api-creditscore.herokuapp.com/shap_client/"
#API_PRED = "http://127.0.0.1:8000/predict/"
#API_SHAP = "http://127.0.0.1:8000/shap_client/"

data = joblib.load('sample_test_set.pickle')
infos_client = joblib.load('infos_client.pickle')
pret_client = joblib.load('pret_client.pickle')
preprocessed_data = joblib.load('preprocessed_data.pickle')
model = joblib.load('model.pkl')

column_names = preprocessed_data.columns.tolist()
expected_value = -2.9159221699244515

classifier = model.named_steps['classifier']
df_preprocess = model.named_steps['preprocessor'].transform(data)
explainer = shap.TreeExplainer(classifier)
generic_shap = explainer.shap_values(df_preprocess, check_additivity=False)

html="""           
    <h1 style="font-size:300%; color:DarkSlateGrey; font-family:Soleil"> Prêt à dépenser <br>
        <h2 style="color:LightSlateGrey; font-family:Sofia Pro"> Tableau de bord</h2>
     </h1>
     <body style="font-size:100%; color:DarkSlateGrey; font-family:Sofia Pro"> <br>
     </body>
"""
st.markdown(html, unsafe_allow_html=True)

#---------#
# SIDEBAR #
#---------#

#Profile Client
profile_ID = st.sidebar.selectbox('Sélectionnez un client :',
                                  list(data.index))
API_GET = API_PRED+(str(profile_ID))
score_client = re.get(API_GET).json()
if score_client > 0.5:
    st.sidebar.write("Le prêt n'est pas octroyé.")
else:
    st.sidebar.write("Le prêt est octroyé.")

# Affichage de la jauge
fig_jauge = go.Figure(go.Indicator(
                      mode='gauge+number+delta',
                      value=score_client,
                      domain={'x': [0, 1], 'y': [0, 1]},
                      gauge={'axis': {'range': [None, 100],
                                      'tickwidth': 3,
                                      'tickcolor': 'black'},
                             'bar': {'color': 'white', 'thickness': 0.25},
                             'steps': [{'range': [0, 25], 'color': 'Green'},
                                       {'range': [25, 49.49], 'color': 'LimeGreen'},
                                       {'range': [49.5, 50.5], 'color': 'red'},
                                       {'range': [50.51, 75], 'color': 'Orange'},
                                       {'range': [75, 100], 'color': 'Crimson'}],
                             'threshold': {'line': {'color': 'white', 'width': 10},
                                           'thickness': 0.5,
                                           'value': np.median(score_client).astype('float')}}))

fig_jauge.update_layout(height=250, width=305,
                        font={'color': 'black', 'family': 'Sofia Pro'},
                        margin=dict(l=0, r=0, b=0, t=0, pad=2))

st.sidebar.plotly_chart(fig_jauge)
if 0 <= score_client < .25:
    score_text = 'Crédit score : EXCELLENT'
    st.sidebar.success(score_text)
elif .25 <= score_client < .50:
    score_text = 'Crédit score : BON'
    st.sidebar.success(score_text)
elif .50 <= score_client < .75:
    score_text = 'Crédit score : MOYEN'
    st.sidebar.warning(score_text)
else :
    score_text = 'Crédit score : BAS'
    st.sidebar.error(score_text)

#---------------------------------------#
# INFORMATIONS GÉNÉRIQUES SUR LE CLIENT #
#---------------------------------------#

# Infos principales client
st.write("Récapitulatif du profil")
client_info = infos_client[infos_client.index == profile_ID].iloc[:, :]
st.table(client_info)

# Infos principales sur la demande de prêt
st.write("Caractéristiques du prêt")
client_pret = pret_client[pret_client.index== profile_ID].iloc[:, :]
st.table(client_pret)

#Explicabilité
API_GET = API_SHAP+(str(profile_ID))
shap_values = re.get(API_GET).json()
shap_values = np.array(shap_values[1:-1].split(',')).astype('float32')
waterfall = shap.plots._waterfall.waterfall_legacy(expected_value=expected_value,
                                                   shap_values=shap_values,
                                                   feature_names=column_names)


#st.pyplot(forceplot)
st.pyplot(waterfall)

#-----------------------#
# CRÉATION DE VARIABLES #
#-----------------------#

# Choix du mode de fonctionnement
mode = st.selectbox('Choisissez le mode',
                    options = ['Graphiques interactifs',
                               'Interprétabilité globale'],
                    index=1)

# Mode graphique interactifs
if mode ==  'Graphiques interactifs' :
    features = st.multiselect("Choisissez deux variables", list(data.columns))
    if len(features) != 2 :
        st.error("Sélectionnez deux variables")
    else :
        st.write("## Graphique interactif avec le défaut attendu en couleur")
        # Graphique
        chart = px.scatter(data, x=features[0], y=features[1], color='TARGET')
        st.plotly_chart(chart)
    st.button("Recommencer")

# Mode interprétabilité globale
if mode == 'Interprétabilité globale':
    st.write('Le graphique suivant indique les variables ayant le plus contribué au modèle.')
    #shap.plots.bar(dict(shap_values), max_display=40)
    summary = shap.summary_plot(generic_shap,
                                column_names)
    st.pyplot(summary)

    st.button("Recommencer")