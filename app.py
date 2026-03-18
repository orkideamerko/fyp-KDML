from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)
models = {}

def load_all_models():
    global models
    try:
        models['Universal'] = {
            'model': joblib.load('models/hybrid/Universal_Base_Model.pkl'),
            'scaler': joblib.load('models/hybrid/Universal_Scaler.pkl'),
            'label_encoder': joblib.load('models/hybrid/Universal_Label_Encoder.pkl'),
            'accuracy': 0.7842
        }
    except: pass
    
    try:
        with open('models/language_specific/model_info.json', 'r') as f:
            model_info = json.load(f)
        for m in model_info['models']:
            nat_name = m['file']
            models[m['name']] = {
                'model': joblib.load(f'models/language_specific/{nat_name}_model.pkl'),
                'scaler': joblib.load(f'models/language_specific/{nat_name}_scaler.pkl'),
                'label_encoder': joblib.load(f'models/language_specific/{nat_name}_label_encoder.pkl'),
                'accuracy': m['accuracy']
            }
    except: pass

def extract_features(keystroke_data):
    dwell_times = [k.get('dwellTime', 0) for k in keystroke_data if k.get('dwellTime')]
    flight_times = [k.get('flightTime', 0) for k in keystroke_data if k.get('flightTime')]
    
    if len(dwell_times) < 10:
        return None
    
    dwell_times = np.array(dwell_times)
    flight_times = np.array(flight_times) if flight_times else np.array([80]*len(dwell_times))
    
    return {
        'dwell1_mean': np.mean(dwell_times), 'dwell1_std': np.std(dwell_times),
        'flight_DD_mean': np.mean(flight_times), 'flight_DD_std': np.std(flight_times),
        'dwell1_median': np.median(dwell_times), 'flight_DD_median': np.median(flight_times),
        'dwell1_iqr': np.percentile(dwell_times, 75) - np.percentile(dwell_times, 25),
        'flight_DD_iqr': np.percentile(flight_times, 75) - np.percentile(flight_times, 25),
        'dwell_cv': np.std(dwell_times) / np.mean(dwell_times),
        'flight_cv': np.std(flight_times) / np.mean(flight_times),
        'dwell_flight_ratio': np.mean(dwell_times) / np.mean(flight_times),
        'dwell2_mean': np.mean(dwell_times), 'flight_UD_mean': np.mean(flight_times),
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/authenticate', methods=['POST'])
def authenticate():
    try:
        keystrokes = request.json.get('keystrokes', [])
        features = extract_features(keystrokes)
        if not features:
            return jsonify({'error': 'Need 10+ keystrokes'}), 400
        
        # Test only top 3 models for speed
        results = {}
        feature_df = pd.DataFrame([features])
        
        for name in list(models.keys())[:3]:  # Only test 3 models for speed
            model_data = models[name]
            X = model_data['scaler'].transform(feature_df)
            proba = model_data['model'].predict_proba(X)[0]
            
            results[name] = {
                'confidence': float(np.max(proba)),
                'predicted_user': f"User_{np.argmax(proba)}"
            }
        
        best = max(results.items(), key=lambda x: x[1]['confidence'])
        return jsonify({'success': True, 'results': results, 'best_model': best[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_all_models()
    print(f"\n✅ Loaded {len(models)} models")
    print("📱 Open: http://localhost:8080\n")
    app.run(debug=False, host='0.0.0.0', port=8080)
