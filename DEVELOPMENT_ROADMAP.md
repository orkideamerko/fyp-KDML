# Multilingual Keystroke Dynamics Authentication System
## Development Roadmap & Setup Guide

### Project Overview
**Student:** Orkidea Merko (001342816)  
**Supervisor:** Georgia Sakellari  
**Goal:** Develop a multilingual keystroke dynamics authentication system with EER < 1% and Accuracy > 90%

---

## 🎯 PHASE 1: Environment Setup (Week 1)

### Step 1.1: Install Python and Create Virtual Environment

```bash
# Install Python 3.9+ (if not already installed)
# Download from python.org

# Create project directory
mkdir keystroke-auth-system
cd keystroke-auth-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 1.2: Install Required Libraries

Create a `requirements.txt` file with these packages:

```
# Data processing
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1

# Machine Learning
scikit-learn==1.3.0
lightgbm==4.0.0
xgboost==1.7.6

# Deep Learning (optional for later)
# tensorflow==2.13.0
# torch==2.0.1

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# Data handling
openpyxl==3.1.2
xlsxwriter==3.1.2

# Utilities
jupyter==1.0.0
tqdm==4.65.0
```

Install all packages:
```bash
pip install -r requirements.txt
```

### Step 1.3: Create Project Structure

```
keystroke-auth-system/
│
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and processed data
│   └── features/               # Extracted features
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load and parse datasets
│   ├── preprocessing.py        # Clean and normalize data
│   ├── feature_extraction.py   # Extract keystroke features
│   ├── models.py               # ML model implementations
│   ├── evaluation.py           # Performance metrics
│   └── utils.py                # Helper functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── models/
│   └── saved_models/           # Trained model files
│
├── results/
│   ├── metrics/                # Performance metrics
│   ├── visualizations/         # Plots and charts
│   └── reports/                # Generated reports
│
├── tests/
│   └── test_*.py               # Unit tests
│
├── config/
│   └── config.yaml             # Configuration settings
│
├── requirements.txt
├── README.md
└── main.py                     # Main execution script
```

---

## 🎯 PHASE 2: Dataset Acquisition & Exploration (Week 1-2)

### Step 2.1: Download Multilingual Keystroke Dataset

**Recommended Datasets:**

1. **Buffalo Dataset** (English)
   - URL: https://www.cs.cmu.edu/~keystroke/

2. **BKSD - Bilingual Keystroke Dataset** (English + Arabic)
   - URL: Search GitHub for "bilingual keystroke dynamics dataset"
   - This is mentioned in your project proposal

3. **Alternative multilingual datasets:**
   - Search for "keystroke dynamics dataset" on GitHub
   - Look for datasets with multiple languages

### Step 2.2: Initial Data Exploration

Create a Jupyter notebook to explore the data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/raw/keystroke_data.csv')

# Basic exploration
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Check language distribution
print(df['language'].value_counts())

# Visualize basic patterns
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
df['dwell_time'].hist(bins=50)
plt.title('Dwell Time Distribution')

plt.subplot(1, 3, 2)
df['flight_time'].hist(bins=50)
plt.title('Flight Time Distribution')

plt.subplot(1, 3, 3)
df['language'].value_counts().plot(kind='bar')
plt.title('Language Distribution')
plt.tight_layout()
plt.show()
```

---

## 🎯 PHASE 3: Data Preprocessing (Week 2)

### Step 3.1: Data Cleaning

```python
# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class KeystrokePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def remove_outliers(self, df, columns, threshold=3):
        """Remove outliers using Z-score method"""
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < threshold]
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        # Remove rows with missing critical features
        df = df.dropna(subset=['dwell_time', 'flight_time'])
        return df
    
    def normalize_features(self, df, feature_columns):
        """Normalize timing features"""
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        return df
    
    def filter_by_language(self, df, language):
        """Filter dataset by language"""
        return df[df['language'] == language].copy()
    
    def preprocess(self, df):
        """Complete preprocessing pipeline"""
        # 1. Handle missing values
        df = self.handle_missing_values(df)
        
        # 2. Remove outliers
        timing_cols = ['dwell_time', 'flight_time']
        df = self.remove_outliers(df, timing_cols)
        
        # 3. Normalize features (save for later after feature extraction)
        
        return df
```

---

## 🎯 PHASE 4: Feature Extraction (Week 2-3)

### Step 4.1: Extract Keystroke Features

```python
# src/feature_extraction.py

import pandas as pd
import numpy as np

class KeystrokeFeatureExtractor:
    """Extract timing-based keystroke features"""
    
    def extract_dwell_time(self, key_press_time, key_release_time):
        """Time key is held down"""
        return key_release_time - key_press_time
    
    def extract_flight_time(self, current_press, previous_release):
        """Time between consecutive key events"""
        return current_press - previous_release
    
    def extract_digraph_features(self, df):
        """Extract two-key sequence timings"""
        features = []
        
        # Group by user and session
        for (user, session), group in df.groupby(['user_id', 'session_id']):
            group = group.sort_values('timestamp')
            
            # Calculate digraph latencies
            for i in range(len(group) - 1):
                digraph_time = group.iloc[i+1]['press_time'] - group.iloc[i]['press_time']
                features.append({
                    'user_id': user,
                    'session_id': session,
                    'digraph_latency': digraph_time,
                    'language': group.iloc[i]['language']
                })
        
        return pd.DataFrame(features)
    
    def extract_trigraph_features(self, df):
        """Extract three-key sequence timings"""
        features = []
        
        for (user, session), group in df.groupby(['user_id', 'session_id']):
            group = group.sort_values('timestamp')
            
            for i in range(len(group) - 2):
                trigraph_time = group.iloc[i+2]['press_time'] - group.iloc[i]['press_time']
                features.append({
                    'user_id': user,
                    'session_id': session,
                    'trigraph_latency': trigraph_time,
                    'language': group.iloc[i]['language']
                })
        
        return pd.DataFrame(features)
    
    def extract_statistical_features(self, df):
        """Extract statistical features per user session"""
        features = []
        
        for (user, session), group in df.groupby(['user_id', 'session_id']):
            feature_dict = {
                'user_id': user,
                'session_id': session,
                'language': group['language'].iloc[0],
                
                # Dwell time statistics
                'dwell_mean': group['dwell_time'].mean(),
                'dwell_std': group['dwell_time'].std(),
                'dwell_median': group['dwell_time'].median(),
                'dwell_min': group['dwell_time'].min(),
                'dwell_max': group['dwell_time'].max(),
                
                # Flight time statistics
                'flight_mean': group['flight_time'].mean(),
                'flight_std': group['flight_time'].std(),
                'flight_median': group['flight_time'].median(),
                'flight_min': group['flight_time'].min(),
                'flight_max': group['flight_time'].max(),
                
                # Typing speed
                'typing_speed': len(group) / (group['timestamp'].max() - group['timestamp'].min())
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def extract_all_features(self, df):
        """Extract complete feature set"""
        # Statistical features
        stat_features = self.extract_statistical_features(df)
        
        # Digraph features
        digraph_features = self.extract_digraph_features(df)
        digraph_agg = digraph_features.groupby(['user_id', 'session_id']).agg({
            'digraph_latency': ['mean', 'std', 'median']
        }).reset_index()
        digraph_agg.columns = ['user_id', 'session_id', 'digraph_mean', 'digraph_std', 'digraph_median']
        
        # Merge features
        features = stat_features.merge(digraph_agg, on=['user_id', 'session_id'], how='left')
        
        return features
```

---

## 🎯 PHASE 5: Model Development (Week 3-4)

### Step 5.1: Implement Base Models

```python
# src/models.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import joblib

class KeystrokeAuthenticator:
    """Keystroke dynamics authentication models"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ML model"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif self.model_type == 'lgbm':
            return LGBMClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        return self.model.predict_proba(X_test)
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
```

---

## 🎯 PHASE 6: Evaluation Framework (Week 4-5)

### Step 6.1: Implement Biometric Metrics

```python
# src/evaluation.py

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

class BiometricEvaluator:
    """Evaluate keystroke authentication performance"""
    
    def calculate_far_frr(self, y_true, y_pred_proba, threshold=0.5):
        """Calculate False Acceptance Rate and False Rejection Rate"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # FAR: Impostors accepted / Total impostor attempts
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # FRR: Genuine users rejected / Total genuine attempts
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return far, frr
    
    def calculate_eer(self, y_true, y_pred_proba):
        """Calculate Equal Error Rate"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        frr = 1 - tpr
        
        # Find where FAR = FRR
        eer_threshold = thresholds[np.nanargmin(np.absolute((frr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((frr - fpr)))]
        
        return eer, eer_threshold
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba):
        """Complete evaluation"""
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # FAR and FRR at default threshold
        far, frr = self.calculate_far_frr(y_true, y_pred_proba[:, 1])
        
        # EER
        eer, eer_threshold = self.calculate_eer(y_true, y_pred_proba[:, 1])
        
        results = {
            'accuracy': accuracy,
            'far': far,
            'frr': frr,
            'eer': eer,
            'eer_threshold': eer_threshold
        }
        
        return results
    
    def plot_performance(self, results_dict):
        """Visualize performance across languages/models"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        models = list(results_dict.keys())
        accuracies = [results_dict[m]['accuracy'] for m in models]
        eers = [results_dict[m]['eer'] for m in models]
        fars = [results_dict[m]['far'] for m in models]
        frrs = [results_dict[m]['frr'] for m in models]
        
        # Accuracy comparison
        axes[0].bar(models, accuracies)
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].axhline(y=0.90, color='r', linestyle='--', label='Target: 90%')
        axes[0].legend()
        
        # EER comparison
        axes[1].bar(models, eers)
        axes[1].set_title('Equal Error Rate Comparison')
        axes[1].set_ylabel('EER (%)')
        axes[1].axhline(y=0.01, color='r', linestyle='--', label='Target: 1%')
        axes[1].legend()
        
        # FAR vs FRR
        x = np.arange(len(models))
        width = 0.35
        axes[2].bar(x - width/2, fars, width, label='FAR')
        axes[2].bar(x + width/2, frrs, width, label='FRR')
        axes[2].set_title('FAR vs FRR')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models)
        axes[2].legend()
        
        plt.tight_layout()
        return fig
```

---

## 🎯 PHASE 7: Main Execution Pipeline (Week 5)

### Step 7.1: Create Main Script

```python
# main.py

import pandas as pd
from src.preprocessing import KeystrokePreprocessor
from src.feature_extraction import KeystrokeFeatureExtractor
from src.models import KeystrokeAuthenticator
from src.evaluation import BiometricEvaluator
from sklearn.model_selection import train_test_split

def main():
    print("=" * 60)
    print("Multilingual Keystroke Dynamics Authentication System")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv('data/raw/keystroke_data.csv')
    print(f"Loaded {len(df)} samples from {df['user_id'].nunique()} users")
    
    # 2. Preprocess
    print("\n[2/6] Preprocessing data...")
    preprocessor = KeystrokePreprocessor()
    df_clean = preprocessor.preprocess(df)
    print(f"After preprocessing: {len(df_clean)} samples")
    
    # 3. Extract features
    print("\n[3/6] Extracting features...")
    extractor = KeystrokeFeatureExtractor()
    features = extractor.extract_all_features(df_clean)
    print(f"Extracted {features.shape[1]} features")
    
    # 4. Prepare data for training
    print("\n[4/6] Preparing training data...")
    X = features.drop(['user_id', 'session_id', 'language'], axis=1)
    y = features['user_id']  # Binary: genuine user vs impostor
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 5. Train models
    print("\n[5/6] Training models...")
    models = ['random_forest', 'svm', 'lgbm']
    results = {}
    
    for model_type in models:
        print(f"\n  Training {model_type}...")
        model = KeystrokeAuthenticator(model_type=model_type)
        model.train(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Evaluate
        evaluator = BiometricEvaluator()
        results[model_type] = evaluator.evaluate_model(y_test, y_pred, y_pred_proba)
        
        print(f"    Accuracy: {results[model_type]['accuracy']:.2%}")
        print(f"    EER: {results[model_type]['eer']:.2%}")
    
    # 6. Display results
    print("\n[6/6] Final Results:")
    print("\n" + "=" * 60)
    for model, metrics in results.items():
        print(f"\n{model.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  EER: {metrics['eer']:.4f}")
        print(f"  FAR: {metrics['far']:.4f}")
        print(f"  FRR: {metrics['frr']:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
```

---

## 📋 QUICK START CHECKLIST

- [ ] Install Python 3.9+
- [ ] Create virtual environment
- [ ] Install required packages
- [ ] Create project structure
- [ ] Download multilingual dataset
- [ ] Run initial data exploration
- [ ] Implement preprocessing
- [ ] Implement feature extraction
- [ ] Implement models
- [ ] Implement evaluation
- [ ] Run main pipeline
- [ ] Document results

---

## 🚀 NEXT STEPS

1. **Week 1-2:** Setup + Dataset acquisition
2. **Week 2-3:** Feature engineering
3. **Week 3-4:** Model training
4. **Week 4-5:** Cross-language evaluation
5. **Week 5-6:** Results analysis + Documentation

---

## 💡 TIPS

- Start simple: Get one model working with one language first
- Test incrementally: Don't write all code before testing
- Use Jupyter notebooks for exploration and visualization
- Document your experiments and findings
- Version control with Git (highly recommended)
- Keep track of all hyperparameters and results

---

Good luck with your project! 🎓
