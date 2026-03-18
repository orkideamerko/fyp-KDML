"""
Biometric Security Metrics Analysis for Keystroke Authentication
Calculates industry-standard security metrics: EER, FAR, FRR, ROC curves

Author: Orkidea Merko
Project: COMP1682 Final Year Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print(" " * 15 + "🔐 BIOMETRIC SECURITY METRICS ANALYSIS 🔐")
print(" " * 20 + "EER, FAR, FRR, ROC Curves")
print("="*80 + "\n")

# Create directories
os.makedirs('results/biometric_security', exist_ok=True)

# ============================================================================
# STEP 1: LOAD TRAINED MODEL AND RECREATE TEST DATA
# ============================================================================
print("[1/6] Loading trained model and recreating test data...")

# Load the universal model
model = joblib.load('models/hybrid/Universal_Base_Model.pkl')
scaler = joblib.load('models/hybrid/Universal_Scaler.pkl')
label_encoder = joblib.load('models/hybrid/Universal_Label_Encoder.pkl')

print("✓ Loaded universal model (78.42% accuracy)")

# We need to recreate the session data with the SAME features the model was trained on
# Load raw data
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

# Get top 5 nationalities
nationality_counts = df['nationality'].value_counts()
top_5_nationalities = nationality_counts.head(5).index.tolist()

print(f"✓ Processing {len(top_5_nationalities)} nationalities")

# Recreate sessions with HYBRID features (matching the universal model)
SESSION_SIZE = 150

def extract_hybrid_features(session_keystrokes):
    """Extract features matching the universal model"""
    
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

all_sessions = []

for nationality in top_5_nationalities:
    nationality_data = df[df['nationality'] == nationality].copy()
    participant_counts = nationality_data['participant'].value_counts()
    valid_participants = participant_counts[participant_counts >= 600].head(15).index
    nationality_data = nationality_data[nationality_data['participant'].isin(valid_participants)].copy()
    
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
            features['session_id'] = f"{participant}_{session_idx}"
            
            all_sessions.append(features)

sessions_df = pd.DataFrame(all_sessions)

print(f"✓ Created {len(sessions_df)} sessions")

# Prepare features
feature_cols = [col for col in sessions_df.columns if col not in ['participant', 'nationality', 'session_id']]
X = sessions_df[feature_cols]
y = sessions_df['participant']

# Encode and scale
y_encoded = label_encoder.transform(y)
X_scaled = scaler.transform(X)

# Get probability predictions
y_proba = model.predict_proba(X_scaled)

print(f"✓ Total sessions: {len(X)}")
print(f"✓ Total users: {len(np.unique(y_encoded))}")

# ============================================================================
# STEP 2: CALCULATE GENUINE AND IMPOSTOR SCORES
# ============================================================================
print("\n[2/6] Calculating genuine and impostor scores...")

genuine_scores = []
impostor_scores = []

# For each session
for i in range(len(X)):
    true_user = y_encoded[i]
    probabilities = y_proba[i]
    
    # Genuine score: probability of correct user
    genuine_score = probabilities[true_user]
    genuine_scores.append(genuine_score)
    
    # Impostor scores: probabilities of all other users
    for j, prob in enumerate(probabilities):
        if j != true_user:
            impostor_scores.append(prob)

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

print(f"\n✓ Genuine attempts: {len(genuine_scores)}")
print(f"✓ Impostor attempts: {len(impostor_scores)}")
print(f"✓ Average genuine score: {np.mean(genuine_scores):.4f}")
print(f"✓ Average impostor score: {np.mean(impostor_scores):.4f}")

# ============================================================================
# STEP 3: CALCULATE FAR, FRR, AND EER
# ============================================================================
print("\n[3/6] Calculating FAR, FRR, and EER...")

# Create binary labels: 1 = genuine, 0 = impostor
y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
y_scores = np.concatenate([genuine_scores, impostor_scores])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# FAR = FPR (False Positive Rate)
# FRR = 1 - TPR (False Negative Rate)
far = fpr
frr = 1 - tpr

# Calculate EER (Equal Error Rate) - where FAR = FRR
eer_threshold_idx = np.nanargmin(np.absolute((far - frr)))
eer = far[eer_threshold_idx]
eer_threshold = thresholds[eer_threshold_idx]

# Calculate AUC (Area Under Curve)
roc_auc = auc(fpr, tpr)

print("\n" + "="*80)
print("BIOMETRIC SECURITY METRICS")
print("="*80)
print(f"EER (Equal Error Rate):     {eer:.4f} ({eer*100:.2f}%)")
print(f"EER Threshold:              {eer_threshold:.4f}")
print(f"AUC (Area Under Curve):     {roc_auc:.4f}")
print("="*80)

# Calculate FAR and FRR at different thresholds
key_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_results = []

for threshold in key_thresholds:
    idx = np.argmin(np.abs(thresholds - threshold))
    
    far_at_threshold = far[idx]
    frr_at_threshold = frr[idx]
    
    threshold_results.append({
        'Threshold': threshold,
        'FAR': far_at_threshold,
        'FRR': frr_at_threshold,
        'FAR_percent': far_at_threshold * 100,
        'FRR_percent': frr_at_threshold * 100
    })

threshold_df = pd.DataFrame(threshold_results)

print("\n" + "="*80)
print("FAR/FRR AT DIFFERENT THRESHOLDS")
print("="*80)
print(threshold_df.to_string(index=False))
print("="*80)

# ============================================================================
# STEP 4: FIND OPTIMAL OPERATING POINTS
# ============================================================================
print("\n[4/6] Finding optimal operating points...")

# 1. EER point
eer_point = {
    'name': 'EER Point (FAR = FRR)',
    'threshold': eer_threshold,
    'FAR': eer,
    'FRR': eer
}

# 2. High Security Point
high_security_idx = np.argmin(np.abs(far - 0.01))
high_security_point = {
    'name': 'High Security (FAR ≈ 1%)',
    'threshold': thresholds[high_security_idx],
    'FAR': far[high_security_idx],
    'FRR': frr[high_security_idx]
}

# 3. Balanced Point
balanced_idx = np.argmin(np.abs(far - 0.05))
balanced_point = {
    'name': 'Balanced (FAR ≈ 5%)',
    'threshold': thresholds[balanced_idx],
    'FAR': far[balanced_idx],
    'FRR': frr[balanced_idx]
}

# 4. High Convenience Point
convenience_idx = np.argmin(np.abs(frr - 0.05))
convenience_point = {
    'name': 'High Convenience (FRR ≈ 5%)',
    'threshold': thresholds[convenience_idx],
    'FAR': far[convenience_idx],
    'FRR': frr[convenience_idx]
}

operating_points = [eer_point, high_security_point, balanced_point, convenience_point]

print("\n" + "="*80)
print("RECOMMENDED OPERATING POINTS")
print("="*80)
for point in operating_points:
    print(f"\n{point['name']}:")
    print(f"  Threshold: {point['threshold']:.4f}")