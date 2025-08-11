import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger objets
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")
encoded_columns = joblib.load("encoded_columns.pkl")
fill_values = joblib.load("fill_values.pkl")
mrg_mapping = joblib.load("mrg_mapping.pkl")

def convert_to_float(value):
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return value

# Champs numériques
montant = st.number_input("Montant", min_value=0.0, step=1.0)
frequence_rech = st.number_input("Fréquence Recharge", min_value=0, step=1)
revenue = st.number_input("Revenue", min_value=0.0, step=1.0)
arpu_segment = st.number_input("ARPU Segment", min_value=0.0, step=1.0)
frequence = st.number_input("Fréquence", min_value=0, step=1)
data_volume = st.number_input("Data Volume", min_value=0.0, step=1.0)
on_net = st.number_input("On Net", min_value=0, step=1)
orange = st.number_input("Orange", min_value=0, step=1)
tigo = st.number_input("Tigo", min_value=0, step=1)
zone1 = st.number_input("Zone1", min_value=0, step=1)
zone2 = st.number_input("Zone2", min_value=0, step=1)
regularity = st.number_input("Regularity", min_value=0, step=1)
freq_top_pack = st.number_input("Freq Top Pack", min_value=0, step=1)

# Champs catégoriels
region = st.selectbox("Region", ["Region1", "Region2", "Region3"])
top_pack = st.selectbox("Top Pack", ["Pack1", "Pack2", "Pack3"])
tenure = st.selectbox("Tenure", ["Court", "Moyen", "Long"])
mrg = st.selectbox("MRG", ["NO", "YES"])

# Bouton prédire
if st.button("Prédire"):
    # Création du DataFrame utilisateur
    X_input = pd.DataFrame([{
        "MONTANT": convert_to_float(montant),
        "FREQUENCE_RECH": convert_to_float(frequence_rech),
        "REVENUE": convert_to_float(revenue),
        "ARPU_SEGMENT": convert_to_float(arpu_segment),
        "FREQUENCE": convert_to_float(frequence),
        "DATA_VOLUME": convert_to_float(data_volume),
        "ON_NET": convert_to_float(on_net),
        "ORANGE": convert_to_float(orange),
        "TIGO": convert_to_float(tigo),
        "ZONE1": convert_to_float(zone1),
        "ZONE2": convert_to_float(zone2),
        "REGULARITY": convert_to_float(regularity),
        "FREQ_TOP_PACK": convert_to_float(freq_top_pack),
        "REGION": region,
        "TOP_PACK": top_pack,
        "TENURE": tenure,
        "MRG": mrg 
    }])

    # Remplir NaN avec mêmes valeurs
    for col, val in fill_values.items():
        if col in X_input.columns:
            X_input[col].fillna(val, inplace=True)

    # Encodage binaire MRG (si besoin, sinon la ligne précédente suffit)
    X_input["MRG"] = X_input["MRG"].map(mrg_mapping)

    # Encodage One-Hot
    X_input_encoded = pd.get_dummies(X_input, columns=cat_cols, drop_first=True)

    # Réaligner colonnes comme à l’entraînement
    for col in encoded_columns:
        if col not in X_input_encoded.columns:
            X_input_encoded[col] = 0

    # Garder l’ordre exact des colonnes
    X_input_encoded = X_input_encoded[encoded_columns]

    ####Verification NAN
    print("NANNNNNNNNNNN")
    print(X_input_encoded.isnull().sum())
    print(X_input_encoded.isna().sum()[X_input_encoded.isna().sum() > 0])


    # Normalisation
    X_input_scaled = scaler.transform(X_input_encoded)

  ####Verification NAN
    print("OOOOOOOOOOOOOOOOOOO")
    print(np.isnan(X_input_scaled).sum())

    # Prédiction
    pred = model.predict(X_input_scaled)[0]
    proba = model.predict_proba(X_input_scaled)[0][1]

    # Affichage résultat
    st.write(f"**Prédiction :** {'Churn' if pred == 1 else 'Pas churn'}")
    st.write(f"**Probabilité de churn :** {proba:.2%}")
