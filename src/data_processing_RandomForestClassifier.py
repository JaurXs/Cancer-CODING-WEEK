# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement de la dataset
data = pd.read_csv("cervical_cancer.csv")

# Afficher les premières lignes
print(data.head())

# Informations sur les colonnes et les types de données
print(data.info())

# Statistiques générales
print(data.describe())

# Remplacer les valeurs manquantes par la médiane
data = data.replace('?', np.nan)
data = data.fillna(data.median())

# Variables explicatives
X = data.drop("Biopsy", axis=1)

# Variable cible
y = data["Biopsy"]

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
