from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Wczytanie modeli i skalera
rf_model = joblib.load("models/rf.pkl")
knn_model = joblib.load("models/knn.pkl")
svm_model = joblib.load("models/svm.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def predict():
    predictions = {}
    vol = 0.7
    sulfur = 223.0
    chlor = 0.3
    sulph = 1.0
    
    if request.method == "POST":
        vol = float(request.form["volatile_acidity"])
        sulfur = float(request.form["total_sulfur_dioxide"])
        chlor = float(request.form["chlorides"])
        sulph = float(request.form["sulphates"])

        X = scaler.transform([[vol, sulfur, chlor, sulph]])
        for name, model in {
            "Random Forest": rf_model,
            "KNN": knn_model,
            "SVM": svm_model
        }.items():
            result = model.predict(X)[0]
            predictions[name] = "Czerwone" if result == 1 else "Białe"

    return render_template("form.html", predictions=predictions, vol=vol, sulfur=sulfur, chlor=chlor, sulph=sulph)


if __name__ == "__main__":
    app.run(debug=True)
