"""
Train Language-Specific Models for Multi-Model Web Application
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings
import os
import json
warnings.filterwarnings('ignore')

print("Training language-specific models...")

# Create directories
os.makedirs('models/language_specific', exist_ok=True)

# Load data (same as before)
demographics = pd.read_csv('data/raw/demographics.csv')
free_text = pd.read_csv('data/raw/free-text.csv')
df = free_text.merge(demographics, on='participant', how='left')

column_mapping = {
    'DU.key1.key1': 'dwell_key1',
    'DU.key1.key2': 'dwell_key2',
    'DD.key1.key2': 'flight_time_DD',
    'UD.key1.key2': 'flight_time_UD'
}
df = df.rename(columns=column_mapping)

for col in ['dwell_key1', 'dwell_key2', 'flight_time_DD', 'flight_time_UD']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['dwell_key1', 'flight_time_DD'])

top_5_nationalities = df['nationality'].value_counts().head(5).index.tolist()

SESSION_SIZE = 150

def extract_hybrid_features(session_keystrokes):
    abs_feat = {
        'dwell1_mean': session_keystrokes['dwell_key1'].mean(),
        'dwell1_std': session_keystrokes['dwell_key1'].std(),
        'flight_DD_mean': session_keystrokes['flight_time_DD'].mean(),
        'flight_DD_std': session_keystrokes['flight_time_DD'].std(),
    }
    robust_feat = {
        'dwell1_median': session_keystrokes['dwell_key1'].median(),
        'flight_DD_median': session_keystrokes['flight_time_DD'].median(),
        'dwell1_iqr': session_keystrokes['dwell_key1'].quantile(0.75) - session_keystrokes['dwell_key1'].quantile(0.25),
        'flight_DD_iqr': session_keystrokes['flight_time_DD'].quantile(0.75) - session_keystrokes['flight_time_DD'].quantile(0.25),
    }
    ratio_feat = {
        'dwell_cv': abs_feat['dwell1_std'] / abs_feat['dwell1_mean'] if abs_feat['dwell1_mean'] > 0 else 0,
        'flight_cv': abs_feat['flight_DD_std'] / abs_feat['flight_DD_mean'] if abs_feat['flight_DD_mean'] > 0 else 0,
        'dwell_flight_ratio': abs_feat['dwell1_mean'] / abs_feat['flight_DD_mean'] if abs_feat['flight_DD_mean'] > 0 else 0,
    }
    context_feat = {
        'dwell2_mean': session_keystrokes['dwell_key2'].mean(),
        'flight_UD_mean': session_keystrokes['flight_time_UD'].mean(),
    }
    return {**abs_feat, **robust_feat, **ratio_feat, **context_feat}

model_results = []

for nationality in top_5_nationalities:
    print(f"Training {nationality} model...")
    nationality_data = df[df['nationality'] == nationality].copy()
    participant_counts = nationality_data['participant'].value_counts()
    valid_participants = participant_counts[participant_counts >= 600].head(15).index
    nationality_data = nationality_data[nationality_data['participant'].isin(valid_participants)].copy()
    
    sessions = []
    for participant in valid_participants:
        user_data = nationality_data[nationality_data['participant'] == participant].copy()
        user_data = user_data.sort_values('session').reset_index(drop=True)
        n_sessions = len(user_data) // SESSION_SIZE
        
        for session_idx in range(n_sessions):
            start_idx = session_idx * SESSION_SIZE
            end_idx = start_idx + SESSION_SIZE
            session_keystrokes = user_data.iloc[start_idx:end_idx]
            features = extract_hybrid_features(session_keystrokes)
            features['participant'] = participant
            features['nationality'] = nationality
            sessions.append(features)
    
    if len(sessions) < 50:
        continue
    
    sessions_df = pd.DataFrame(sessions)
    feature_cols = [col for col in sessions_df.columns if col not in ['participant', 'nationality']]
    X = sessions_df[feature_cols]
    y = sessions_df['participant']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, max_features='sqrt', random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    model_name = f"{nationality.replace(' ', '_')}"
    joblib.dump(model, f'models/language_specific/{model_name}_model.pkl')
    joblib.dump(scaler, f'models/language_specific/{model_name}_scaler.pkl')
    joblib.dump(label_encoder, f'models/language_specific/{model_name}_label_encoder.pkl')
    
    print(f"  ✓ {nationality}: {accuracy*100:.2f}% accuracy ({len(sessions)} sessions)")
    
    model_results.append({
        'name': nationality,
        'file': model_name,
        'accuracy': float(accuracy),
        'users': len(valid_participants),
        'sessions': len(sessions)
    })

model_info = {'models': model_results}
with open('models/language_specific/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\n✅ Trained {len(model_results)} models successfully!")
