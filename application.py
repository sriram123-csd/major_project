from flask import Flask, render_template, request, redirect, session
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)
app.secret_key = 'colorectal_cancer_prediction_2026'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "artifacts", "models", "random_forest_model.pkl")

model = joblib.load(model_path)

# Feature columns (in the same order as training)
FEATURE_COLUMNS = [
    'Country', 'Age', 'Gender', 'Cancer_Stage', 'Tumor_Size_mm',
    'Family_History', 'Smoking_History', 'Alcohol_Consumption', 'Obesity_BMI',
    'Diet_Risk', 'Physical_Activity', 'Diabetes', 'Inflammatory_Bowel_Disease',
    'Genetic_Mutation', 'Screening_History', 'Early_Detection', 'Treatment_Type',
    'Healthcare_Costs', 'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K',
    'Urban_or_Rural', 'Economic_Classification', 'Healthcare_Access', 'Insurance_Status'
]

# Define fixed encoding mappings to match training data
ENCODING_MAP = {
    'Country': {'USA': 0, 'Canada': 1, 'UK': 2, 'India': 3, 'Australia': 4},
    'Gender': {'M': 1, 'F': 0, 'Male': 1, 'Female': 0},
    'Cancer_Stage': {'Localized': 0, 'Regional': 1, 'Metastatic': 2},
    'Family_History': {'Yes': 1, 'No': 0},
    'Smoking_History': {'Yes': 1, 'No': 0},
    'Alcohol_Consumption': {'Yes': 1, 'No': 0},
    'Obesity_BMI': {'Normal': 0, 'Overweight': 1, 'Obese': 2},
    'Diet_Risk': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Physical_Activity': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Diabetes': {'Yes': 1, 'No': 0},
    'Inflammatory_Bowel_Disease': {'Yes': 1, 'No': 0},
    'Genetic_Mutation': {'Yes': 1, 'No': 0},
    'Screening_History': {'Regular': 1, 'Irregular': 0, 'Never': 2, 'Occasional': 0},
    'Early_Detection': {'Yes': 1, 'No': 0},
    'Treatment_Type': {'Surgery': 0, 'Chemotherapy': 1, 'Radiotherapy': 2, 'Combination': 3, 'Radiation': 2, 'Combined': 3},
    'Urban_or_Rural': {'Urban': 1, 'Rural': 0},
    'Economic_Classification': {'Developed': 1, 'Developing': 0},
    'Healthcare_Access': {'High': 2, 'Moderate': 1, 'Low': 0},
    'Insurance_Status': {'Insured': 1, 'Uninsured': 0}
}

def get_recommendations(age, gender, cancer_stage, family_history, smoking, alcohol, diet_risk, physical_activity, diabetes):
    """Generate personalized recommendations based on patient data"""
    
    precautions = []
    diet_recommendations = []
    exercise_recommendations = []
    lifestyle_tips = []
    medical_advice = ""

    # Precautions
    precautions.append("Schedule regular colorectal cancer screenings as recommended by your doctor")
    
    if cancer_stage == "Regional" or cancer_stage == "Metastatic":
        precautions.append("Seek immediate consultation with an oncologist")
    
    if family_history == "Yes":
        precautions.append("Inform all first-degree relatives about increased family risk")
    
    if smoking == "Yes":
        precautions.append("Quit smoking immediately - smoking increases cancer risk by 30-40%")
    
    if alcohol == "Yes":
        precautions.append("Limit or eliminate alcohol consumption")
    
    if diabetes == "Yes":
        precautions.append("Maintain strict blood sugar control and monitor regularly")
    
    if int(age) > 50:
        precautions.append("Perform annual medical check-ups")

    # Diet Recommendations
    diet_recommendations.append("Increase fiber intake to 25-35g daily (whole grains, vegetables, fruits)")
    diet_recommendations.append("Consume at least 5 servings of fruits and vegetables daily")
    
    if diet_risk == "High":
        diet_recommendations.append("Reduce red and processed meat consumption to less than 18oz/week")
        diet_recommendations.append("Avoid sugary drinks and excessive processed foods")
    
    diet_recommendations.append("Include omega-3 rich foods (fish, flaxseeds, walnuts) twice weekly")
    diet_recommendations.append("Add fermented foods (yogurt, kefir) for gut health")
    
    if diabetes == "Yes":
        diet_recommendations.append("Follow diabetic nutritionist's meal plan with low glycemic index foods")

    # Exercise Recommendations
    if physical_activity == "Low":
        exercise_recommendations.append("Start with 30 minutes of moderate walking daily (150 min/week)")
        exercise_recommendations.append("Gradually increase to 300 minutes of moderate activity per week")
    elif physical_activity == "Moderate":
        exercise_recommendations.append("Continue 150-300 minutes of moderate aerobic activity weekly")
        exercise_recommendations.append("Add strength training 2-3 times per week")
    else:
        exercise_recommendations.append("Maintain current exercise routine of 300+ minutes weekly")
        exercise_recommendations.append("Incorporate high-intensity interval training 2 days/week")
    
    exercise_recommendations.append("Include core and flexibility exercises (yoga, pilates) 2x weekly")
    exercise_recommendations.append("Avoid prolonged sitting - stand and move every 30 minutes")

    # Lifestyle Tips
    lifestyle_tips.append("Maintain healthy BMI (18.5-24.9) through balanced diet and exercise")
    lifestyle_tips.append("Get 7-9 hours of quality sleep every night")
    lifestyle_tips.append("Reduce stress through meditation, yoga, or mindfulness (20 min daily)")
    lifestyle_tips.append("Stay hydrated - drink at least 8 glasses of water daily")
    
    if smoking == "Yes":
        lifestyle_tips.append("Enroll in a smoking cessation program for professional support")
    
    lifestyle_tips.append("Maintain regular bowel movements - don't ignore urges")
    lifestyle_tips.append("Have regular health check-ups with preventive screening tests")

    # Medical Advice
    if cancer_stage == "Localized":
        medical_advice = "Your cancer is in early stage. With appropriate treatment and lifestyle changes, survival rates are excellent (>90%). Follow your doctor's treatment plan strictly and maintain regular follow-ups every 3 months."
    elif cancer_stage == "Regional":
        medical_advice = "Your cancer has spread to nearby lymph nodes. Multi-modal treatment (surgery + chemotherapy/radiation) is often recommended. Prognosis depends on treatment response. Regular monitoring is essential."
    elif cancer_stage == "Metastatic":
        medical_advice = "Your cancer has spread to distant organs. This requires aggressive treatment combining chemotherapy, targeted therapy, or immunotherapy. Consult with your oncology team for personalized treatment options. Clinical trials may also be available."
    
    if age > 70:
        medical_advice += " Given your age, discuss treatment options carefully with your doctor considering overall health status and life expectancy."

    return precautions, diet_recommendations, exercise_recommendations, lifestyle_tips, medical_advice

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form inputs
        country_input = request.form.get("country", "USA").strip()
        
        # Normalize country input to match ENCODING_MAP
        country_map = {'USA': 'USA', 'us': 'USA', 'Canada': 'Canada', 'ca': 'Canada', 'UK': 'UK', 'gb': 'UK', 'India': 'India', 'in': 'India', 'Australia': 'Australia', 'au': 'Australia'}
        country = country_map.get(country_input, 'USA')
        
        data = {
            'Country': country,
            'Age': float(request.form.get("age")),
            'Gender': request.form.get("gender"),
            'Cancer_Stage': request.form.get("cancer_stage"),
            'Tumor_Size_mm': float(request.form.get("tumor_size_mm")),
            'Family_History': request.form.get("family_history"),
            'Smoking_History': request.form.get("smoking_history"),
            'Alcohol_Consumption': request.form.get("alcohol_consumption"),
            'Obesity_BMI': request.form.get("obesity_bmi"),
            'Diet_Risk': request.form.get("diet_risk"),
            'Physical_Activity': request.form.get("physical_activity"),
            'Diabetes': request.form.get("diabetes"),
            'Inflammatory_Bowel_Disease': request.form.get("ibd"),
            'Genetic_Mutation': request.form.get("genetic_mutation"),
            'Screening_History': request.form.get("screening_history"),
            'Early_Detection': request.form.get("early_detection"),
            'Treatment_Type': request.form.get("treatment_type"),
            'Healthcare_Costs': float(request.form.get("healthcare_costs")),
            'Incidence_Rate_per_100K': float(request.form.get("incidence_rate")),
            'Mortality_Rate_per_100K': float(request.form.get("mortality_rate")),
            'Urban_or_Rural': 'Urban',  # Default value
            'Economic_Classification': 'Developed',  # Default value
            'Healthcare_Access': 'Moderate',  # Default value
            'Insurance_Status': 'Insured'  # Default value
        }

        # Create DataFrame with features in correct order
        input_df = pd.DataFrame([data])
        
        # Ensure all features are in order
        input_df = input_df[FEATURE_COLUMNS]

        # Encode categorical features using fixed encoding map (consistent with training)
        for col in input_df.columns:
            if col in ENCODING_MAP:
                # Use the predefined encoding map for consistency
                input_df[col] = input_df[col].map(lambda x: ENCODING_MAP[col].get(str(x).strip(), 0))
            elif col not in ['Age', 'Tumor_Size_mm', 'Healthcare_Costs', 'Incidence_Rate_per_100K', 'Mortality_Rate_per_100K']:
                # For any other categorical columns not in map
                le = LabelEncoder()
                input_df[col] = le.fit_transform(input_df[col].astype(str))

        # Convert to correct dtype
        input_df = input_df.astype(np.float32)

        # Random Forest doesn't need scaling, make prediction directly
        prediction = model.predict(input_df)[0]
        
        # Invert the prediction (model was trained with opposite encoding)
        # So 1 from model = LOW risk, 0 from model = HIGH risk
        risk_level = 'LOW' if prediction == 1 else 'HIGH'
        
        # Store results in session
        session['risk_level'] = risk_level
        session['age'] = int(data['Age'])
        session['gender'] = data['Gender']
        session['cancer_stage'] = data['Cancer_Stage']
        session['family_history'] = data['Family_History']
        session['smoking'] = data['Smoking_History']
        session['alcohol'] = data['Alcohol_Consumption']
        session['diet_risk'] = data['Diet_Risk']
        session['physical_activity'] = data['Physical_Activity']
        session['diabetes'] = data['Diabetes']

        return redirect("/results")

    except Exception as e:
        error_msg = f"Error processing form: {str(e)}"
        print(f"PREDICTION ERROR: {error_msg}")  # Print to terminal for debugging
        return render_template("form.html", error=error_msg)

@app.route("/results")
def results():
    risk_level = session.get('risk_level', 'UNKNOWN')
    
    # Generate recommendations
    precautions, diet_recs, exercise_recs, lifestyle_tips, medical_advice = get_recommendations(
        session.get('age', 50),
        session.get('gender', 'M'),
        session.get('cancer_stage', 'Localized'),
        session.get('family_history', 'No'),
        session.get('smoking', 'No'),
        session.get('alcohol', 'No'),
        session.get('diet_risk', 'Low'),
        session.get('physical_activity', 'Moderate'),
        session.get('diabetes', 'No')
    )
    
    if risk_level == 'HIGH':
        risk_message = "Your assessment indicates increased risk. Please consult with a healthcare professional immediately for further evaluation and treatment options."
    else:
        risk_message = "Your assessment indicates lower risk. However, continue healthy habits and regular screening as recommended. Prevention is always better than cure."

    return render_template("results.html",
                         risk_level=risk_level,
                         risk_message=risk_message,
                         precautions=precautions,
                         diet_recommendations=diet_recs,
                         exercise_recommendations=exercise_recs,
                         lifestyle_tips=lifestyle_tips,
                         medical_advice=medical_advice)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
