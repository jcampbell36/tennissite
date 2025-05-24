from flask import Flask, request, render_template
import pickle, os

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")  # lets you swap models later
app = Flask(__name__)
model = pickle.load(open(MODEL_PATH, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    x = float(request.form["x"])
    return str(model.predict([[x]])[0])
