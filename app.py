
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model/placement_model.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict page
@app.route("/predict")
def predict():
    return render_template("predict.html")

# Result
@app.route("/result", methods=["POST"])
def result():
    try:
        cgpa = float(request.form.get("cgpa") or 0)
        internships = float(request.form.get("internships") or 0)
        projects = float(request.form.get("projects") or 0)
        workshops = float(request.form.get("workshops") or 0)
        aptitude = float(request.form.get("aptitude") or 0)
        soft = float(request.form.get("soft") or 0)
        extracurricular = float(request.form.get("extra") or 0)
        training = float(request.form.get("training") or 0)
        ssc = float(request.form.get("ssc") or 0)
        hsc = float(request.form.get("hsc") or 0)

        data = np.array([[cgpa, internships, projects, workshops,
                          aptitude, soft, extracurricular,
                          training, ssc, hsc]])

        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1] * 100

        result_text = "Placed ✅" if prediction == 1 else "Not Placed ❌"

        return render_template("result.html",
                               result=result_text,
                               probability=round(prob,2))

    except Exception as e:
        return f"ERROR: {e}"

if __name__ == "__main__":
    app.run(debug=True)