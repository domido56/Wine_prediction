from flask import Flask, render_template, request
from ml.loader import get_full_dataset, load_models, get_test_data, features, target
from ml.visualizer import cechy, plot_conf_matrix, plot_feature_importance, generate_classification_report

app = Flask(__name__)

rf_model, knn_model, svm_model, scaler = load_models()
model_dict = {
    "random_forest": rf_model,
    "knn": knn_model,
    "svm": svm_model
}

@app.route("/", methods=["GET", "POST"])
def predict():
    predictions = {}
    vol = 0.7
    sulfur = 223.0
    chlor = 0.3
    sulph = 1.0
    image_base64 = None
    
    if request.method == "POST":
        vol = float(request.form["volatile_acidity"])
        sulfur = float(request.form["total_sulfur_dioxide"])
        chlor = float(request.form["chlorides"])
        sulph = float(request.form["sulphates"])

        image_base64 = cechy(vol, sulfur, chlor, sulph)

        X = scaler.transform([[vol, sulfur, chlor, sulph]])
        for name, model in {
            "Random Forest": rf_model,
            "KNN": knn_model,
            "SVM": svm_model
        }.items():
            result = model.predict(X)[0]
            predictions[name] = "Czerwone" if result == 1 else "Białe"

    return render_template("form.html", predictions=predictions, vol=vol, sulfur=sulfur, chlor=chlor, sulph=sulph,
                           scatter_plot=image_base64)

@app.route("/<model_name>")
def model_detail(model_name):
    model = model_dict.get(model_name)
    if not model:
        return "Model nieznany", 404
    
    X_test, y_test = get_test_data()
    y_pred = model.predict(X_test)

    cm_img = plot_conf_matrix(y_test, y_pred)
    fi_img = plot_feature_importance(model, features)
    report = generate_classification_report(y_test, y_pred)

    return render_template("model_detail.html", 
                           model_name=model_name.upper(), 
                           report=report,
                           confusion_matrix_img=cm_img,
                           feature_importance_img=fi_img)


if __name__ == "__main__":
    app.run(debug=True)
