from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved models
reg_model = joblib.load("Projects/student_pridiction/models/best_regression_Ridge.pkl")   # Example file name
cls_model = joblib.load("Projects/student_pridiction/models/best_classifier_GaussianNB.pkl")

FEATURES = ["Attendance","StudyHours","PastScore","SleepHours",
            "Assignments","InternetUse","LabScore",
            "ParentEdu","ParentIncome","Tuition"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = []

    for f in FEATURES:
        data.append(float(request.form[f]))

    data = np.array(data).reshape(1, -1)

    final_score = reg_model.predict(data)[0]
    pass_fail = cls_model.predict(data)[0]

    status = "PASS" if pass_fail == 1 else "FAIL"

    return render_template("result.html",
                            score=final_score,
                            status=status)

if __name__ == "__main__":
    app.run(debug=True)
