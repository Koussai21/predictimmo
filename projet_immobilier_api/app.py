from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

"""
API Flask pour prédiction de prix immobilier à Paris
Endpoint: POST /predict - Prédire le prix au m² d'un bien immobilier
"""

app = Flask(__name__)

# ============================================
# CONFIGURATION ET CHARGEMENT DES MODÈLES
# ============================================

# Chemins relatifs aux fichiers modèles
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Charger le modèle XGBoost et le scaler
with open(os.path.join(BASE_DIR, 'modele_xgb_baseline.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'scaler_baseline.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Colonnes attendues (dans le bon ordre)
FEATURE_COLUMNS = [
    'surface_reelle_bati', 'surface_terrain', 'nombre_pieces_principales',
    'nombre_lots', 'surface_totale', 'annee', 'mois', 'arrondissement',
    'code_postal', 'distance_center_km', 'latitude', 'longitude',
    'type_local', 'nature_mutation'
]

# Valeurs possibles pour les colonnes catégoriques
VALID_TYPE_LOCAL = ['Appartement', 'Maison', 'Terrain', 'Local commercial']
VALID_NATURE_MUTATION = ['Vente', 'Échange', 'Apport', 'Partage', 'Expropriation', 'Dation']


# ============================================
# ROUTES
# ============================================

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil avec documentation"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction
    
    Entrée (JSON):
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
    
    Sortie (JSON):
    {
        "success": true,
        "prediction_log": 10.5234,
        "prediction_eur_m2": 50234,
        "surface": 85.5
    }
    """
    
    try:
        # Récupérer les données JSON
        data = request.get_json()
        
        if data is None:
            return jsonify({
                'success': False,
                'error': 'Données JSON invalides'
            }), 400
        
        # Valider que toutes les colonnes sont présentes
        missing_columns = [col for col in FEATURE_COLUMNS if col not in data]
        if missing_columns:
            return jsonify({
                'success': False,
                'error': f'Colonnes manquantes: {", ".join(missing_columns)}'
            }), 400
        
        # Valider les valeurs catégoriques
        if data['type_local'] not in VALID_TYPE_LOCAL:
            return jsonify({
                'success': False,
                'error': f"type_local invalide. Valeurs acceptées: {VALID_TYPE_LOCAL}"
            }), 400
        
        if data['nature_mutation'] not in VALID_NATURE_MUTATION:
            return jsonify({
                'success': False,
                'error': f"nature_mutation invalide. Valeurs acceptées: {VALID_NATURE_MUTATION}"
            }), 400
        
        # Créer un DataFrame avec une seule ligne
        df_input = pd.DataFrame([data])[FEATURE_COLUMNS]
        
        # Sauvegarder la surface pour le calcul final
        surface_m2 = float(data['surface_reelle_bati'])
        
        # ===== PREPROCESSING =====
        # 1. Remplacer les valeurs NaN/null par des valeurs par défaut
        df_input['surface_terrain'] = df_input['surface_terrain'].fillna(0.0)
        
        # 2. Encoder les variables catégoriques (correspondence avec le modèle)
        type_local_mapping = {
            'Appartement': 0,
            'Maison': 1,
            'Terrain': 2,
            'Local commercial': 3
        }
        nature_mutation_mapping = {
            'Vente': 0,
            'Échange': 1,
            'Apport': 2,
            'Partage': 3,
            'Expropriation': 4,
            'Dation': 5
        }
        
        # 3. Récupérer SEULEMENT les 12 colonnes numériques (avant d'encoder les catégories)
        X_numeric = df_input.iloc[:, :12].values  # Colonnes 0-11 uniquement
        
        # 4. Appliquer le scaler (il attend 12 features)
        X_scaled = scaler.transform(X_numeric)
        
        # 5. Faire la prédiction avec les données normalisées
        prediction_log = model.predict(X_scaled)[0]
        
        # 6. Convertir de log(prix_m2) à prix en EUR/m²
        prediction_eur_m2 = np.exp(prediction_log)
        
        # Retourner le résultat
        return jsonify({
            'success': True,
            'prediction_log': round(float(prediction_log), 4),
            'prediction_eur_m2': int(round(prediction_eur_m2)),
            'surface_m2': surface_m2,
            'prix_total_estime': int(round(prediction_eur_m2 * surface_m2))
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Endpoint pour vérifier que l'API fonctionne"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }), 200


# ============================================
# LANCEMENT DE L'APP
# ============================================

if __name__ == '__main__':
    print("="*70)
    print("DÉMARRAGE DE L'API FLASK")
    print("="*70)
    print("\n URL: http://127.0.0.1:5000")
    print("   - Page d'accueil      : GET  http://127.0.0.1:5000/")
    print("   - Prédiction          : POST http://127.0.0.1:5000/predict")
    print("   - Santé de l'API      : GET  http://127.0.0.1:5000/health")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
