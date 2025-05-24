from flask import Flask, request, render_template
import joblib, os

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")   # or "model.joblib"
app = Flask(__name__)
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    x = float(request.form["x"])
    proba = model.predict_proba([[x]])[0, 1]
    return f"{proba:.4f}"

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000   # default 8000
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)
