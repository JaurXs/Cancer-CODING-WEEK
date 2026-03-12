# La prédiction nécesite ces bibliothèques et 1 à 6
import numpy as np
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Chargement de la dataset-1
data = pd.read_csv("data.csv")

# Remplacer les valeurs manquantes par la médiane-2
data = data.replace('?', np.nan)
data = data.fillna(data.median())

# Variables explicatives-3
X = data.drop("Biopsy", axis=1)

# Variable cible-4
y = data["Biopsy"]

# Séparation des données-5
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

# Création du modèle-6
model = RandomForestClassifier(n_estimators=200,random_state=42)

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Rapport complet
print(classification_report(y_test, y_pred))

# Importance des variables
importances = model.feature_importances_

feature_importance = pd.DataFrame({ "Feature": X.columns,"Importance": importances})
    
print(feature_importance.sort_values(by="Importance", ascending=False))


