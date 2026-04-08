import pandas as pd
import numpy as np

DATA_PATH = "data/bronze"

fichiers = [
    "75_2020.csv",
    "75_2021.csv",
    "75_2022.csv",
    "75_2023.csv",
    "75_2024.csv",
    "75_2025.csv"
]

liste_df = []
for f in fichiers:
    chemin = f"{DATA_PATH}/{f}" if DATA_PATH != "." else f
    print(f"Chargement de : {f}")
    temp = pd.read_csv(chemin, low_memory=False)
    liste_df.append(temp)

df = pd.concat(liste_df, ignore_index=True)
print(f"Données combinées : {df.shape[0]:,} lignes et {df.shape[1]} colonnes\n")

print("=== Étape 2 : Nettoyage des données ===")

df_clean = df.copy()

df_clean = df_clean[df_clean['code_departement'] == 75]
print(f"après filtre Paris: {df_clean.shape[0]:,} lignes")

df_clean = df_clean.dropna(subset=['valeur_fonciere', 'surface_reelle_bati'])
print(f"Après suppression des valeurs manquantes prix/surface : {df_clean.shape[0]:,} lignes")

df_clean = df_clean[df_clean['valeur_fonciere'] > 0]
print(f"Après suppression prix <= 0 : {df_clean.shape[0]:,} lignes")

df_clean = df_clean[df_clean['surface_reelle_bati'] > 10]
print(f"Après suppression surfaces <= 10 m² : {df_clean.shape[0]:,} lignes")


df_clean['prix_m2'] = df_clean['valeur_fonciere'] / df_clean['surface_reelle_bati']

print("\n=== Statistiques du prix au m² après nettoyage ===")
print(df_clean['prix_m2'].describe().round(2))

print("\n5 prix au m² les plus bas :")
print(df_clean['prix_m2'].nsmallest(5).round(2))

print("\n5 prix au m² les plus hauts :")
print(df_clean['prix_m2'].nlargest(5).round(2))

print("\n=== Étape 3 : Sauvegarde du fichier nettoyé (couche silver) ===")

output_file = "data/silver/dvf_75_nettoye.csv"

df_clean.to_csv(output_file, index=False, encoding='utf-8')

print(f"Fichier sauvegardé avec succès : {output_file}")
print(f"   → {df_clean.shape[0]:,} lignes")
print(f"   → {df_clean.shape[1]} colonnes")