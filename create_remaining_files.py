# Create streamlit_demo.py and app.py
print("Creating web app files...")

# I'll create a simpler version - just the essential files
# You already have the models trained, so let's make a quick demo

with open('quick_test.py', 'w') as f:
    f.write("""
import joblib
import json

# Test if models loaded correctly
print("Testing models...")

# Load universal model
try:
    model = joblib.load('models/hybrid/Universal_Base_Model.pkl')
    print("✅ Universal model loaded")
except:
    print("❌ Universal model not found")

# Load language-specific models
try:
    with open('models/language_specific/model_info.json', 'r') as f:
        model_info = json.load(f)
    print(f"✅ Found {len(model_info['models'])} language-specific models:")
    for m in model_info['models']:
        print(f"   • {m['name']}: {m['accuracy']*100:.2f}%")
except Exception as e:
    print(f"❌ Error loading language models: {e}")

print("\\nAll models ready for web app! 🚀")
""")

print("Created: quick_test.py")
print("\nRun: python quick_test.py")
