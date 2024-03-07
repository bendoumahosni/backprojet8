#!/usr/bin/env python
# coding: utf-8

# In[14]:


from fastapi import FastAPI, Path
from pydantic import BaseModel
from typing import Optional
from joblib import load
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import shap


# In[15]:


# Importez du modèle 

model = load('lgbm_w.joblib')

app = FastAPI()
# scale de donnees 
scaler = MinMaxScaler(feature_range = (0, 1))

# importation des donnees
df=pd.read_csv('./test_app.csv')
df=df[:50] 
# Classe Input /output
class ClientInput(BaseModel):
    client_id: int

class PredictionOutput(BaseModel):
    client_id: int
    predicted_class: int
    predicted_score: float
   
class InfoCLient(BaseModel):
    client_id : int
    ext_source_1 : float
    ext_source_2 : float
    ext_source_3 : float
    good_price : float
    payment_rate : float
    amt_annuity : float
    days_birth : int
    code_gender : int
    amt_credit : float
    days_employed : int

@app.get("/predict/{client_id}")
async def predict_class(client_id: int = Path(..., title="ID du client")):
    # recuperation des caracteristiques du dataframe
    features_for_client_id = get_features_for_client_id(client_id)
    
    if features_for_client_id is not None:
        
        predicted_proba = model.predict_proba([list(features_for_client_id.values())])[0]
        predicted_class = int(model.predict([list(features_for_client_id.values())])[0] > 0.499)  
        predicted_score = float(predicted_proba[1])
        # la classe de sortie
        output = PredictionOutput(client_id=client_id, predicted_class=predicted_class, predicted_score=predicted_score)
        return output
    else:
        return {"error": "Client non trouvé"}

# Fonction pour récupérer les caractéristiques du client
def get_features_for_client_id(client_id):
    df=pd.read_csv('./test_app.csv')
    df=df[:50]
    # recherche du client 
    client_data = df[df['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR'])
    if client_data.shape[0]==1:
        scaler.fit(client_data)
        scaled_client_data = scaler.transform(client_data)
        features = client_data.columns.tolist()  # Obtenez les noms des colonnes
        return dict(zip(features, client_data.values[0]))
    else:
        return None

@app.get('/get_info/{client_id}')
def get_info(client_id: int = Path(..., title="ID du client")):
    features_for_client_id = get_features_for_client_id(client_id)
    
    if features_for_client_id is not None:
        ext_source_1 = features_for_client_id["EXT_SOURCE_1"]
        ext_source_2 = features_for_client_id["EXT_SOURCE_2"]
        ext_source_3 = features_for_client_id["EXT_SOURCE_3"]
        good_price = features_for_client_id['AMT_GOODS_PRICE']
        payment_rate = features_for_client_id['PAYMENT_RATE']
        amt_annuity = features_for_client_id['AMT_ANNUITY']
        days_birth = features_for_client_id['DAYS_BIRTH']
        code_gender = features_for_client_id['CODE_GENDER_F']
        amt_credit = features_for_client_id['AMT_CREDIT']
        days_employed = features_for_client_id['DAYS_EMPLOYED']
        out = InfoCLient(client_id=client_id,ext_source_1=ext_source_1,ext_source_2=ext_source_2,ext_source_3=ext_source_3,good_price=good_price,payment_rate=payment_rate,amt_annuity=amt_annuity,
                         days_birth=days_birth,code_gender=code_gender,amt_credit=amt_credit,days_employed=days_employed)
        return out


#  Etude de l'importance
explainer=shap.Explainer(model)
shap_values=explainer.shap_values(df.drop(columns=['SK_ID_CURR']))
global_feature_importances = {
    'feature_names': list(df.columns),
    'importances': shap_values[0].tolist()
}

# importances globale
@app.get("/get_global_feature_importances")
async def get_global_feature_importances():
    return global_feature_importances

# Importances locales 
@app.get("/get_local_feature_importances/{client_id}")
async def get_local_feature_importances(client_id: int = Path(..., title="ID du client")):
    
   # Récupérer les caractéristiques du client
    features_for_client_id = get_features_for_client_id(client_id)
    if features_for_client_id is not None:
        # Calculer les valeurs SHAP pour le client sélectionné
        shap_values = explainer.shap_values(pd.DataFrame([features_for_client_id]))
        local_feature_importances = {
            'feature_names': list(df.columns),
            'importances': shap_values[0].tolist()
        }
        return local_feature_importances
    else:
        return {"error": "Client non trouvé"}

@app.get('/')
def index():
    return "Bonjour, c'est la page d'index"


# In[ ]:




