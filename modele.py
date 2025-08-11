

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from ydata_profiling import ProfileReport
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc,  accuracy_score, f1_score, roc_auc_score, confusion_matrix,roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import gdown


# Mode interactif ON
plt.ion()

###1. Chargement et aperçu des données

#file_id = "12_KUHr5NlHO_6bN5SylpkxWc-JvpJNWe"
##url = f"https://drive.google.com/uc?id={file_id}"
output = "D:\\Users\\mtirchi\\Downloads\\Expresso_churn_dataset.csv"

# Télécharger le fichier depuis Google Drive
###gdown.download(url, output, quiet=False)

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv(output)

print(df.head())

#Affichez des informations générales sur l'ensemble de données
print(f"Head :")
print(df.head())
print(f"describe :")
print(df.describe())
print(f"Info :")
print(df.info())
print(f"Shape :")
print(df.shape)

print(df.columns)

#Créez un rapport de profilage pandas pour obtenir des informations sur l'ensemble de données
profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_notebook_iframe()


##Le rapport profiling nous a indiqué  
##- l'éxistence des données null environ 35%
##-la variable TENURE est fortement désiquilibrée 86%
##-Plusieurs variables corrélées 

## nombre de données NaN
print(f"valeur null : {df.isna().sum()}")

## nombre des données en double
print(f"Doublons : {df.duplicated().sum()}")

## Supprimer les Doublons
df.drop_duplicates(inplace=True)



# Sauvegarde mapping MRG
mrg_mapping = {'NO': 0, 'YES': 1}

# Sauvegarde valeurs de remplacement NaN
fill_values = {
    "MONTANT": df["MONTANT"].median(),
    "FREQUENCE_RECH": df["FREQUENCE_RECH"].mode()[0],
    "REVENUE": df["REVENUE"].median(),
    "ARPU_SEGMENT": df["ARPU_SEGMENT"].median(),
    "FREQUENCE": df["FREQUENCE"].median(),
    "DATA_VOLUME": df["DATA_VOLUME"].median(),
    "ON_NET": df["ON_NET"].median(),
    "ORANGE": df["ORANGE"].median(),
    "TIGO": df["TIGO"].median(),
    "ZONE1": df["ZONE1"].median(),
    "ZONE2": df["ZONE2"].median(),
    "REGULARITY": df["REGULARITY"].median(),
    "TOP_PACK": df["TOP_PACK"].mode()[0],
    "FREQ_TOP_PACK": df["FREQ_TOP_PACK"].median(),
}


# Application du remplissage NaN
for col, val in fill_values.items():
    df[col].fillna(val, inplace=True)

# Encodage MRG
df['MRG'] = df['MRG'].map(mrg_mapping)


# Sélectionner des variables
X = df.drop(["CHURN", "user_id"], axis=1)
y = df["CHURN"]

# Colonnes catégorielles
cat_cols = ['REGION', 'TOP_PACK', 'TENURE']
# Colonnes numériques
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols = [col for col in num_cols if col not in cat_cols]

# Définir colonnes numériques et catégorielles
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = ['REGION', 'TOP_PACK', 'TENURE']

# Encodage one-hot
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Sauvegarder les noms des colonnes après encodage
encoded_columns = X_encoded.columns.tolist()



print(X_encoded.head())


print(X_encoded.head())

print(df.shape)

print(X_encoded.shape)

# Normalisation
scaler = StandardScaler()
X_encoded_scaled = scaler.fit_transform(X_encoded)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded_scaled, y, test_size=0.2, random_state=42)





### Logistic Regression
model_lr = LogisticRegression(
    solver="saga",           # gère grands volumes + L1/L2
    penalty="l2",            # régularisation L2
    C=1.0,                   # force de régularisation (plus petit => plus régulier)
    class_weight="balanced", # gère le déséquilibre
    max_iter=100,            # itérations
    n_jobs=-1,               # utilise tous les cœurs CPU
    random_state=42
)

model_lr.fit(X_train, y_train)




#  Évaluation

y_pred = model_lr.predict(X_test)
y_proba = model_lr.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy : {acc:.2f}")
print(f"F1-score : {f1:.2f}")
print(f"AUC : {roc_auc:.2f}")



#ROC
y_proba = model_lr.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc="lower right")
plt.show()


#matrice de confusion
y_pred = model_lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Valeurs prédites")
plt.ylabel("Valeurs réelles")
plt.title("Matrice de confusion")
plt.show()

##Accuracy + F1_score


acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f" Accuracy : {acc:.2f}")
print(f" F1-score : {f1:.2f}")
print(f" AUC : {roc_auc:.2f}")




import joblib

joblib.dump(model_lr, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(num_cols, "num_cols.pkl")
joblib.dump(cat_cols, "cat_cols.pkl")
joblib.dump(encoded_columns, "encoded_columns.pkl")  # Nouvelle sauvegarde
joblib.dump(fill_values, "fill_values.pkl")
joblib.dump(mrg_mapping, "mrg_mapping.pkl")