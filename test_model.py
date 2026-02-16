import joblib
import pandas as pd
import numpy as np

model = joblib.load('artifacts/models/random_forest_model.pkl')

# Test with LOW RISK data (all preventive factors)
test_data = {
    'Country': 0,  # USA
    'Age': 30.0,
    'Gender': 1,  # Male
    'Cancer_Stage': 0,  # Localized
    'Tumor_Size_mm': 5.0,
    'Family_History': 0,  # No
    'Smoking_History': 0,  # No
    'Alcohol_Consumption': 0,  # No
    'Obesity_BMI': 0,  # Normal
    'Diet_Risk': 0,  # Low
    'Physical_Activity': 2,  # High
    'Diabetes': 0,  # No
    'Inflammatory_Bowel_Disease': 0,  # No
    'Genetic_Mutation': 0,  # No
    'Screening_History': 1,  # Regular
    'Early_Detection': 1,  # Yes
    'Treatment_Type': 0,  # Surgery
    'Healthcare_Costs': 5000.0,
    'Incidence_Rate_per_100K': 10.0,
    'Mortality_Rate_per_100K': 3.0,
    'Urban_or_Rural': 1,  # Urban
    'Economic_Classification': 1,  # Developed
    'Healthcare_Access': 1,  # Moderate
    'Insurance_Status': 1   # Insured
}

# Convert to DataFrame
df = pd.DataFrame([test_data])
print("✓ Input shape:", df.shape)
print("✓ Input features:", len(df.columns))

# Make prediction
prediction = model.predict(df)[0]
probability = model.predict_proba(df)[0]

print("\n" + "="*50)
print("PREDICTION RESULT:")
print("="*50)
print(f"Risk class: {prediction}")
print(f"Probabilities: Low={probability[0]:.2%}, High={probability[1]:.2%}")
print(f"🟢 Risk Level: {'LOW' if prediction == 0 else '🔴 HIGH'}")
print("="*50)
