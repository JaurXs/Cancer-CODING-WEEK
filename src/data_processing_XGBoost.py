import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# chemin des données
data_dir = Path("data")
raw_path = data_dir / "risk_factors_cervical_cancer.csv"


def load_data():
    """Charge le dataset brut et sépare X et y"""
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {raw_path}")

    df = pd.read_csv(raw_path)

    X = df.drop(columns=["Biopsy"])
    y = df["Biopsy"]

    return X, y


# chemins des fichiers splittés
xtrain_path = data_dir / "X_train_cleaned.csv"
ytrain_path = data_dir / "y_train_cleaned.csv"
xtest_path = data_dir / "X_test_cleaned.csv"
ytest_path = data_dir / "y_test_cleaned.csv"


if xtrain_path.exists() and ytrain_path.exists() and xtest_path.exists() and ytest_path.exists():

    print("ℹ️ Fichiers splittés déjà présents. Chargement...")

    X_train = pd.read_csv(xtrain_path)
    y_train = pd.read_csv(ytrain_path).squeeze()

    X_test = pd.read_csv(xtest_path)
    y_test = pd.read_csv(ytest_path).squeeze()

else:

    print("ℹ️ Chargement du dataset brut")

    X, y = load_data()

    print("ℹ️ Création du train/test split")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train.to_csv(xtrain_path, index=False)
    y_train.to_csv(ytrain_path, index=False)

    X_test.to_csv(xtest_path, index=False)
    y_test.to_csv(ytest_path, index=False)

<<<<<<< HEAD
    print("✅ Données sauvegardées avec succès")
=======
# Sauvegarder les données nettoyées
X_train.to_csv(data_dir / 'X_train_cleaned.csv', index=False)
y_train.to_csv(data_dir / 'y_train_cleaned.csv', index=False)
X_test.to_csv(data_dir / 'X_test_cleaned.csv', index=False)
y_test.to_csv(data_dir / 'y_test_cleaned.csv', index=False)

print("✅ Données nettoyées et sauvegardées avec succès !")

# 4. CHARGEMENT SÉCURISÉ (maintenant que les fichiers existent)
try:
    # On construit le chemin complet vers chaque fichier
    X_train = pd.read_csv(data_dir / 'X_train_cleaned.csv')
    print("✅ X_train chargé avec succès !")
    
    # Si le premier passe, on peut charger les autres
    Y_train = pd.read_csv(data_dir / 'y_train_cleaned.csv').squeeze()
    X_test = pd.read_csv(data_dir / 'X_test_cleaned.csv')
    Y_test = pd.read_csv(data_dir / 'y_test_cleaned.csv').squeeze()
    
    print("\n🚀 TOUT EST PRÊT ! Voici un aperçu des données :")
    print(X_train.head())

except Exception as e:
    print(f"\n💥 ÉCHEC FINAL : {e}")
    
import numpy as np

def optimize_memory(df):

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtype

        if col_type != object:

            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            else:

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2

    print(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")

    return df
>>>>>>> 2362fd5c3113cac898a31869eb1862f8d09f5af8
