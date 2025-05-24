from flask import Flask, request, render_template
import joblib, os

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")   # keep it configurable
app = Flask(__name__)
model = joblib.load(MODEL_PATH)                        # ← joblib, not pickle

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    x = float(request.form["x"])
    return str(model.predict([[x]])[0])

if __name__ == "__main__":
    # disable reloader so the model isn’t loaded twice
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
