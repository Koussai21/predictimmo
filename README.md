# Predictimmo - Prédiction des Prix Immobiliers à Paris

## Contexte du projet

Ce projet vise à développer un modèle de machine learning capable d'estimer les prix des biens immobiliers à Paris sur la base de données réelles de transactions immobilières (données DVF - Demandes de Valeurs Foncières) couvrant la période 2020-2025.

L'objectif est de comprendre les patterns de prix immobiliers à Paris et de fournir un outil de prédiction accessible via une API web pour estimer le prix au m² d'un bien immobilier en fonction de ses caractéristiques (surface, localisation, arrondissement, date, etc.).

## Ce que fait le projet

Le projet fourni un système complet d'analyse et de prédiction immobilière :

1. **Exploration et nettoyage des données** : Analyse des données DVF pour identifier les patterns et les anomalies
2. **Feature engineering** : Création de variables pertinentes pour la modélisation
3. **Modélisation** : Entraînement et comparaison de deux modèles de régression (Régression Linéaire et XGBoost)
4. **API web** : Endpoint REST permettant de faire des prédictions en temps réel
5. **Interface utilisateur** : Page web pour tester le modèle de manière interactive

Les prédictions estimées sont le prix au m² en EUR, qui peut être multiplié par la surface pour obtenir un prix total estimé.

## Démarche choisie

### Phase 1 : Préparation et exploration des données
- Chargement des fichiers DVF (2020-2025)
- Nettoyage des données (valeurs manquantes, outliers, données invalides)
- Filtrage : apartments uniquement, Paris uniquement (codes postaux 75xxx), ventes normales uniquement
- Création de features temporelles (année, mois) et géographiques (distance du centre, latitude, longitude)

### Phase 2 : Feature engineering
- Normalisation des variables numériques
- Encodage des variables catégoriques (type de bien, type de mutation)
- Calcul d'une variable cible transformée en log (log du prix au m²) pour mieux capturer la distribution

### Phase 3 : Modélisation
- Séparation train/test (80/20)
- Entraînement de deux modèles : Régression Linéaire et XGBoost
- Evaluation des performances sur l'ensemble de test
- Sélection du meilleur modèle (XGBoost)

### Phase 4 : Mise en production
- Sauvegarde du modèle et du scaler en pickle
- Développement d'une API Flask
- Déploiement d'une interface web

## Structure des notebooks

### 1. eda.ipynb - Analyse Exploratoire (EDA)

**Rôle** : Comprendre les données brutes et identifier les patterns.

**Analyses menées** :
- Statistiques descriptives des variables (min, max, moyenne, médiane, écart-type)
- Distribution de la variable cible (prix_m2 et valeur_fonciere)
- Étude de la normalité des distributions et nécessité d'une transformation log
- Analyse temporelle : évolution du prix par année (2020-2025)
- Analyse par arrondissement et localisation (latitude/longitude)
- Identification des outliers et anomalies
- Étude des corrélations entre variables

**Résultats** :
- Confirme la nécessité d'une transformation log pour la cible
- Identifie les variables les plus importantes pour les prix
- Détecte un effet arrondissement très fort
- Soulève l'existence d'une multicolinéarité entre surface et nombre de pièces

### 2. Preprocessing_Feature_Engineering.ipynb - Nettoyage et Feature Engineering

**Rôle** : Préparer les données brutes pour la modélisation.

**Processus appliqué** :
- Filtrage des données (ventes normales, appartements, Paris uniquement)
- Suppression des outliers sur le prix au m² (P0.5 et P99.5 pour limiter les extrêmes)
- Gestion des valeurs manquantes :
  - surface_terrain : remplacée par 0 si manquante
  - Suppression des lignes avec valeurs manquantes critiques
- Création de features :
  - surface_totale (nouvelle feature)
  - distance_center_km (distance depuis le centre de Paris)
  - Encodage des variables catégoriques (type_local, nature_mutation)
- Normalisation et scaling des variables numériques
- Création de la variable cible en log : y = log(prix_m2)

**Résultats** :
- Ensemble de données final nettoyé et prêt pour la modélisation
- Fichiers sauvegardés en format CSV pour faciliter le chargement
- Réduction des biais liés aux valeurs aberrantes

### 3. Modelisation_Baseline.ipynb - Modélisation et Évaluation

**Rôle** : Entraîner et évaluer des modèles de prédiction.

**Modèles testés** :

1. **Régression Linéaire** :
   - Modèle de base pour établir une ligne de référence
   - Résultats test :
     - MAE : 5 789 EUR/m²
     - RMSE : 8 352 EUR/m²
     - MAPE : 93.73%
     - R² : 0.198

2. **XGBoost** :
   - Gradient boosting pour capturer les patterns non-linéaires
   - Résultats test :
     - MAE : 2 130 EUR/m²
     - RMSE : 3 358 EUR/m²
     - MAPE : 29.01%
     - R² : 0.245

**Analyses menées** :
- Comparaison des performances des deux modèles
- Analyse de l'importance des features : quelles variables influent le plus ?
- Validation croisée des performances
- Identification des limitations du modèle (R² relatif faible = 24.5% d'explication)

**Résultats** :
- XGBoost surpasse la Régression Linéaire
- Modèle XGBoost sélectionné pour la production
- Les features temporelles (année) et spatiales (arrondissement) dominent
- Les erreurs résiduelles suggèrent l'importance de facteurs non captés dans les données (état du bien, prestige de la rue, rénovation, etc.)

## Architecture du projet

```
predictimmo/
├── data/
│   ├── bronze/              # Données brutes DVF (2020-2025)
│   ├── silver/              # Données nettoyées
│   └── gold/                # Données prêtes pour la modélisation
├── models/                  # Modèles et scalers sauvegardés
│   ├── modele_xgb_baseline.pkl
│   └── scaler_baseline.pkl
├── plots/                   # Visualisations et graphiques
├── results/                 # Résultats des métriques
├── notebooks/               # Notebooks Jupyter
│   ├── eda.ipynb           # Analyse exploratoire
│   ├── Preprocessing_Feature_Engineering.ipynb
│   └── Modelisation_Baseline.ipynb
├── projet_immobilier_api/   # API Flask
│   ├── app.py
│   ├── requirements.txt
│   └── templates/
│       └── index.html
└── README.md
```

## Quels résultats peuvent être obtenus

Le modèle permet d'obtenir les estimations suivantes :

### Estimation du prix au m²
En fournissant les caractéristiques d'un bien (surface, pièces, arrondissement, localisation, date), le modèle estime le prix au m² en EUR.

**Exemple** : Pour un appartement de 85 m² au 5e arrondissement en juin 2023 :
- Estimation : 9 500 EUR/m² (peut varier selon les caractéristiques exactes)
- Prix total estimé : 807 500 EUR

### Limites et interprétation
- Le modèle capture environ 24.5% de la variance des prix (R² = 0.245)
- Erreur moyenne : 2 130 EUR/m² (environ 29% d'erreur relative)
- Le modèle identifie les tendances générales mais ne capture pas tous les détails (état du bien, rénovation, prestige, etc.)
- Les prédictions sont valides pour la période 2020-2025 ; l'extrapolation au-delà est risquée

### Insights
- L'arrondissement est la variable la plus importante pour prédire les prix
- La localisation (latitude/longitude, distance du centre) a un fort impact
- L'année (tendance temporelle) montre l'évolution du marché
- Les appartements ont des patterns de prix distincts des autres types de bien

## Utilisation de l'API

### Installation et démarrage

1. **Prérequis** :
   - Python 3.8+
   - Un environnement virtuel configuré (venv ou conda)

2. **Installation des dépendances** :
   ```bash
   pip install -r projet_immobilier_api/requirements.txt
   ```

3. **Démarrage du serveur** :
   ```bash
   cd projet_immobilier_api
   python app.py
   ```
   
   Le serveur démarre sur `http://127.0.0.1:5000`

### Accès à l'interface web

Ouvrez votre navigateur et allez à : `http://127.0.0.1:5000`

Une page web interactive vous permet de remplir les caractéristiques du bien et d'obtenir une prédiction instantanément.

### Utilisation programmatique (API REST)

**Endpoint** : `POST /predict`

**Exemple de requête** :
```json
{
    "surface_reelle_bati": 85.5,
    "surface_terrain": null,
    "nombre_pieces_principales": 3,
    "nombre_lots": 1,
    "surface_totale": 95.0,
    "annee": 2023,
    "mois": 6,
    "arrondissement": 5,
    "code_postal": 75005,
    "distance_center_km": 2.5,
    "latitude": 48.845,
    "longitude": 2.345,
    "type_local": "Appartement",
    "nature_mutation": "Vente"
}
```

**Exemple de réponse** :
```json
{
    "success": true,
    "prediction_log": 10.5234,
    "prediction_eur_m2": 50234,
    "surface_m2": 85.5,
    "prix_total_estime": 4294905
}
```

### Champs obligatoires

- `surface_reelle_bati` : Surface du bâti en m² (nombre)
- `nombre_pieces_principales` : Nombre de pièces principales (entier)
- `nombre_lots` : Nombre de lots (entier)
- `surface_totale` : Surface totale en m² (nombre)
- `annee` : Année de la transaction (2000-2030)
- `mois` : Mois (1-12)
- `arrondissement` : Arrondissement à Paris (1-20)
- `code_postal` : Code postal format 75xxx (75001-75020)
- `distance_center_km` : Distance du centre en km (nombre)
- `latitude` : Latitude GPS
- `longitude` : Longitude GPS
- `type_local` : Type de bien ('Appartement', 'Maison', 'Terrain', 'Local commercial')
- `nature_mutation` : Type de transaction ('Vente', 'Échange', 'Apport', 'Partage', 'Expropriation', 'Dation')

### Champs optionnels

- `surface_terrain` : Surface terrain en m² (si null, sera défini à 0)

### Vérification de la santé de l'API

**Endpoint** : `GET /health`

Retourne l'état du serveur et confirme que le modèle et le scaler sont chargés.

## Exécution complète du projet

### Pour reproduire les analyses

1. **Exécuter le notebook d'analyse exploratoire** :
   ```bash
   jupyter notebook eda.ipynb
   ```

2. **Exécuter le notebook de préparation des données** :
   ```bash
   jupyter notebook Preprocessing_Feature_Engineering.ipynb
   ```

3. **Exécuter le notebook de modélisation** :
   ```bash
   jupyter notebook Modelisation_Baseline.ipynb
   ```

   Ces notebooks génèrent les modèles dans le dossier `models/` et les résultats dans `results/`

4. **Copier les fichiers de modèle dans le dossier API** :
   ```bash
   cp models/modele_xgb_baseline.pkl projet_immobilier_api/
   cp models/scaler_baseline.pkl projet_immobilier_api/
   ```

5. **Démarrer l'API** :
   ```bash
   cd projet_immobilier_api
   python app.py
   ```

## Technologies utilisées

- **Python 3.8+**
- **Pandas** : Manipulation et analyse des données
- **Scikit-learn** : Preprocessing et métriques
- **XGBoost** : Modèle de gradient boosting
- **Flask** : Framework web pour l'API
- **Numpy** : Calculs numériques
- **Pickle** : Sérialisation des modèles
- **HTML/CSS/JavaScript** : Interface web

## Notes importantes

1. **Les prédictions estimées sont des tendances générales** du marché immobilier parisien, pas des prix réels exacts pour un bien spécifique
2. **Le modèle a un R² modéré** (0.245) : il explique 24.5% de la variance ; les 75.5% restants dépendent de facteurs non disponibles dans les données DVF
3. **L'année dans le modèle** n'est pas un forecast : mettre 2027 ne prédit pas le futur, mais extrapole les tendances 2020-2025
4. **Les données DVF incluent uniquement les transactions enregistrées** : pas les ventes non enregistrées ou les prix de listing
5. **La précision diminue pour les bien atypiques** : le modèle fonctionne mieux pour les appartements standards que pour les propriétés exceptionnelles

## Améliorations futures possibles

- Ajouter plus de features (données externes comme prix des commodités, transports, écoles)
- Fine-tuning des hyperparamètres du modèle XGBoost
- Intégrer des données de transactions plus récentes
- Créer des modèles spécialisés par arrondissement
- Ajouter un intervalle de confiance aux prédictions
- Implémenter un système de validation croisée time-serie
