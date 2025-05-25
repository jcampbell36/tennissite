from flask import Flask, request, render_template
import random
import joblib, os

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
app = Flask(__name__)
model = joblib.load(MODEL_PATH)

def generate_acrostic_poem(name):
    # Dictionary of words for each letter
    word_dict = {
        'a': ['ace', 'advantage', 'aggressive', 'athletic'],
        'b': ['backhand', 'baseline', 'break', 'bounce'],
        'c': ['court', 'champion', 'cross-court', 'challenge'],
        'd': ['deuce', 'doubles', 'drop shot', 'defense'],
        'e': ['energy', 'endurance', 'elite', 'excellence'],
        'f': ['forehand', 'fault', 'fifteen', 'footwork'],
        'g': ['game', 'grand slam', 'grip', 'groundstroke'],
        'h': ['hit', 'hold', 'hustle', 'height'],
        'i': ['intensity', 'instinct', 'inside-out', 'impressive'],
        'j': ['jump', 'juice', 'jolt', 'journey'],
        'k': ['killer serve', 'keen', 'knack', 'knowledge'],
        'l': ['lob', 'line', 'love', 'lead'],
        'm': ['match', 'momentum', 'movement', 'master'],
        'n': ['net', 'nervous', 'natural', 'nimble'],
        'o': ['overhead', 'out', 'offense', 'opponent'],
        'p': ['point', 'power', 'practice', 'precision'],
        'q': ['quick', 'quality', 'quiet', 'quest'],
        'r': ['racket', 'rally', 'return', 'rhythm'],
        's': ['serve', 'spin', 'smash', 'slice'],
        't': ['tennis', 'topspin', 'timing', 'technique'],
        'u': ['unbeatable', 'unforced', 'unstoppable', 'unique'],
        'v': ['volley', 'victory', 'vigor', 'versatile'],
        'w': ['winner', 'warm-up', 'wrist', 'warrior'],
        'x': ['xcellent', 'xceptional', 'xtraordinary', 'xpert'],
        'y': ['yearning', 'yield', 'young', 'youthful'],
        'z': ['zeal', 'zealous', 'zip', 'zone']
    }
    
    poem_lines = []
    name = name.lower()
    
    for letter in name:
        if letter.isalpha():
            words = word_dict.get(letter, [''])
            word = random.choice(words)
            line = f"{letter.upper()}: {word.capitalize()} on the tennis court"
            poem_lines.append(line)
    
    return "\n".join(poem_lines)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    x = float(request.form["x"])
    proba = model.predict_proba([[x]])[0, 1]
    return f"{proba:.4f}"

@app.route("/generate_poem", methods=["POST"])
def generate_poem():
    player_name = request.form["player_name"]
    poem = generate_acrostic_poem(player_name)
    return render_template("index.html", poem=poem)

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    app.run(debug=True, host="0.0.0.0", port=port)
