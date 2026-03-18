from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))
CORS(app)

models = {}

def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        models['Universal'] = {
            'model':   joblib.load(f'{base}/models/hybrid/Universal_Base_Model.pkl'),
            'scaler':  joblib.load(f'{base}/models/hybrid/Universal_Scaler.pkl'),
            'encoder': joblib.load(f'{base}/models/hybrid/Universal_Label_Encoder.pkl'),
            'accuracy': 78.42
        }
        print("✅ Universal model loaded")
    except Exception as e:
        print(f"❌ Universal: {e}")

    try:
        with open(f'{base}/models/language_specific/model_info.json') as f:
            info = json.load(f)
        for m in info['models']:
            name = m['name']
            nat  = m['file']
            try:
                models[name] = {
                    'model':   joblib.load(f'{base}/models/language_specific/{nat}_model.pkl'),
                    'scaler':  joblib.load(f'{base}/models/language_specific/{nat}_scaler.pkl'),
                    'encoder': joblib.load(f'{base}/models/language_specific/{nat}_label_encoder.pkl'),
                    'accuracy': m['accuracy']
                }
                print(f"✅ {name} model loaded")
            except Exception as e:
                print(f"❌ {name}: {e}")
    except Exception as e:
        print(f"❌ Could not load model_info.json: {e}")

load_models()

FEATURE_COLS = [
    'dwell1_mean','dwell1_std','flight_DD_mean','flight_DD_std',
    'dwell1_median','flight_DD_median','dwell1_iqr','flight_DD_iqr',
    'dwell_cv','flight_cv','dwell_flight_ratio','dwell2_mean','flight_UD_mean'
]

def get_confidence(model, X_scaled):
    """
    Use mean tree vote depth as confidence score.
    Each tree in the Random Forest votes — we take the average
    probability across ALL trees for the top class, then normalise
    across models so scores spread meaningfully.
    """
    rf = model['model']
    scaler = model['scaler']

    # Get per-tree predictions from every estimator in the forest
    tree_probas = np.array([
        tree.predict_proba(X_scaled)[0]
        for tree in rf.estimators_
    ])  # shape: (n_trees, n_classes)

    # Mean probability across all trees for the winning class
    mean_probas = tree_probas.mean(axis=0)
    top_class_mean = float(np.max(mean_probas))

    # Variance across trees for the winning class (low variance = more confident)
    top_class_idx = np.argmax(mean_probas)
    variance = float(tree_probas[:, top_class_idx].std())

    # Confidence = mean probability penalised by variance
    # High mean + low variance = very confident
    confidence = top_class_mean * (1 - min(variance * 2, 0.5))

    return confidence

@app.route('/')
def index():
    base = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base, 'index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body received'}), 400

    features = data.get('features', {})
    missing = [f for f in FEATURE_COLS if f not in features]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    X = pd.DataFrame([{col: features[col] for col in FEATURE_COLS}])

    raw_scores = {}
    for name, m in models.items():
        try:
            X_scaled = m['scaler'].transform(X)
            raw_scores[name] = get_confidence(m, X_scaled)
        except Exception as e:
            print(f"Prediction error for {name}: {e}")
            raw_scores[name] = 0.0

    # Normalise scores across models so they spread between 30% and 95%
    values = np.array(list(raw_scores.values()))
    v_min, v_max = values.min(), values.max()

    results = []
    for name, score in raw_scores.items():
        if v_max > v_min:
            # Spread scores across 30–95% range
            normalised = 0.30 + (score - v_min) / (v_max - v_min) * 0.65
        else:
            normalised = 0.60

        results.append({
            'model': name,
            'confidence': round(normalised * 100, 1),
            'accuracy': models[name]['accuracy']
        })

    results.sort(key=lambda x: x['confidence'], reverse=True)
    return jsonify({'results': results})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models_loaded': list(models.keys())})

if __name__ == '__main__':
    print(f"\n✅ {len(models)} models loaded")
    print("🚀 Open in Safari: http://127.0.0.1:5000")
    print("🌐 Health check:   http://127.0.0.1:5000/health\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
