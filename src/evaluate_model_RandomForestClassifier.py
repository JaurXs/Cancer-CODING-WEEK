## L'évaluation du modèle nécessite ces bibliothèques et 1 à 6
# Importation des métriques d'évaluation
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# Chargement de la dataset-1
data = pd.read_csv("cervical_cancer.csv")

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

                      # Evaluer le modèle
# Faire les prédictions avec le modèle entraîné
y_pred = model.predict(X_test)

# Calcul de l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy du modèle :", accuracy)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion :")
print(cm)

# Calcul du score ROC-AUC
y_prob = model.predict_proba(X_test)[:,1]
roc_score = roc_auc_score(y_test, y_prob)

print("\nScore ROC-AUC :", roc_score)