from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model & scaler safely
model_path = os.path.join(BASE_DIR, "model", "placement_model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/result", methods=["POST"])
def result():
    try:
        # Get form data
        cgpa = float(request.form.get("cgpa", 0))
        internships = int(request.form.get("internships", 0))
        projects = int(request.form.get("projects", 0))
        workshops = int(request.form.get("workshops", 0))
        aptitude = float(request.form.get("aptitude", 0))
        soft = float(request.form.get("soft", 0))
        extracurricular = int(request.form.get("extra", 0))
        training = float(request.form.get("training", 0))
        ssc = float(request.form.get("ssc", 0))
        hsc = float(request.form.get("hsc", 0))

        # Create DataFrame
        data = pd.DataFrame([[
            cgpa, internships, projects, workshops, aptitude,
            soft, extracurricular, training, ssc, hsc
        ]], columns=[
            'CGPA',
            'Internships',
            'Projects',
            'Workshops/Certifications',
            'AptitudeTestScore',
            'SoftSkillsRating',
            'ExtracurricularActivities',
            'PlacementTraining',
            'SSC_Marks',
            'HSC_Marks'
        ])

        # Scale data
        data_scaled = scaler.transform(data)

        # Prediction
        prediction = model.predict(data_scaled)[0]

        # Probability (Placed = class 1)
        prob = model.predict_proba(data_scaled)[0][1] * 100

        # Custom threshold logic
        if prob >= 60:
            result_text = "You will be Placed 🎉"
        else:
            result_text = "Not Placed ❌"

        return render_template(
            "result.html",
            result=result_text,
            probability=round(prob, 2)
        )

    except Exception as e:
        return f"ERROR: {str(e)}"

# ------------------ RUN ------------------

if __name__ == "__main__":
    app.run(debug=True)