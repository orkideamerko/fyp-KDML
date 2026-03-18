"""
Train Language-Specific Models for Multi-Model Web Application
Creates individual models for each nationality for comparison

Author: Orkidea Merko
Project: COMP1682 Final Year Project
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
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print(" " * 12 + "🌍 TRAINING LANGUAGE-SPECIFIC MODELS 🌍")
print(" " * 15 + "Individual Models for Each Nationality")
print("="*80 + "\n")

# Create directories
os.makedirs('models/language_specific', exist_ok=True)
os.makedirs('results/language_specific', exist_ok=True)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("[1/3] Loading data...")

demographics = pd.read_csv('data/raw/demographics.csv')
free_text = pd.read_csv('data/raw/free-text.csv')

df = free_text.merge(demographics, on='participant', how='left')

# Rename columns
column_mapping = {
    'DU.key1.key1': 'dwell_key1',
    'DU.key1.key2': 'dwell_key2',
    'DD.key1.key2': 'flight_time_DD',
    'UD.key1.key2': 'flight_time_UD'
}
df = df.rename(columns=column_mapping)

# Convert to numeric
for col in ['dwell_key1', 'dwell_key2', 'flight_time_DD', 'flight_time_UD']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['dwell_key1', 'flight_time_DD'])

print(f"✓ Loaded {len(df):,} keystroke pairs")

# Select top 5 nationalities
nationality_counts = df['nationality'].value_counts()
top_5_nationalities = nationality_counts.head(5).index.tolist()

print(f"\n✓ Training models for: {', '.join(top_5_nationalities)}")

# ============================================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================================

SESSION_SIZE = 150

def extract_hybrid_features(session_keystrokes):
    """Extract hybrid features matching universal model"""
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

# ============================================================================
# TRAIN MODELS FOR EACH NATIONALITY
# ============================================================================
print("\n[2/3] Training language-specific models...\n")

model_results = []

for nationality in top_5_nationalities:
    print(f"Training {nationality} model...")
    
    # Get data for this nationality
    nationality_data = df[df['nationality'] == nationality].copy()
    participant_counts = nationality_data['participant'].value_counts()
    
    # Select users with sufficient data
    valid_participants = participant_counts[participant_counts >= 600].head(15).index
    nationality_data = nationality_data[nationality_data['participant'].isin(valid_participants)].copy()
    
    # Create sessions
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
        print(f"  ⚠️ Not enough data for {nationality}, skipping...")
        continue
    
    sessions_df = pd.DataFrame(sessions)
    
    # Prepare features
    feature_cols = [col for col in sessions_df.columns if col not in ['participant', 'nationality']]
    X = sessions_df[feature_cols]
    y = sessions_df['participant']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model
    model_name = f"{nationality.replace(' ', '_')}"
    joblib.dump(model, f'models/language_specific/{model_name}_model.pkl')
    joblib.dump(scaler, f'models/language_specific/{model_name}_scaler.pkl')
    joblib.dump(label_encoder, f'models/language_specific/{model_name}_label_encoder.pkl')
    
    print(f"  ✓ {nationality}: {accuracy*100:.2f}% accuracy ({len(sessions)} sessions, {len(valid_participants)} users)")
    
    model_results.append({
        'Nationality': nationality,
        'Accuracy': accuracy,
        'Sessions': len(sessions),
        'Users': len(valid_participants),
        'Model_File': f'{model_name}_model.pkl'
    })

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[3/3] Saving results...")

results_df = pd.DataFrame(model_results)
results_df.to_csv('results/language_specific/model_summary.csv', index=False)

print(f"\n✓ Trained {len(model_results)} language-specific models")
print(f"✓ Saved to: models/language_specific/")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" " * 18 + "LANGUAGE-SPECIFIC MODELS SUMMARY")
print("="*80)

print("\n📊 TRAINED MODELS:")
for _, row in results_df.iterrows():
    print(f"  • {row['Nationality']:15s}: {row['Accuracy']*100:.2f}% ({row['Users']} users, {row['Sessions']} sessions)")

print(f"\n💾 MODEL FILES:")
print(f"  Location: models/language_specific/")
for _, row in results_df.iterrows():
    nat_name = row['Nationality'].replace(' ', '_')
    print(f"  • {nat_name}_model.pkl")
    print(f"  • {nat_name}_scaler.pkl")
    print(f"  • {nat_name}_label_encoder.pkl")

print(f"\n🌍 READY FOR WEB APPLICATION!")
print(f"  These models will be used to demonstrate:")
print(f"  • Language-specific authentication (high accuracy)")
print(f"  • Cross-nationality testing (shows degradation)")
print(f"  • Universal model comparison (robust alternative)")

print("\n" + "="*80)
print("✅ LANGUAGE-SPECIFIC MODEL TRAINING COMPLETE!")
print("="*80 + "\n")

# Create model info file for web app
model_info = {
    'models': []
}

for _, row in results_df.iterrows():
    nat_name = row['Nationality'].replace(' ', '_')
    model_info['models'].append({
        'name': row['Nationality'],
        'file': nat_name,
        'accuracy': float(row['Accuracy']),
        'users': int(row['Users']),
        'sessions': int(row['Sessions'])
    })

import json
with open('models/language_specific/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✓ Created model_info.json for web application")