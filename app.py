from flask import Flask, request, render_template
import random
import joblib, os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
app = Flask(__name__)
model = joblib.load(MODEL_PATH)

# Initialize Gemini
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
    try:
        # List available models
        for m in genai.list_models():
            print(f"Available model: {m.name}")
        model_gemini = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        print("Successfully initialized Gemini 2.5 Flash Preview")
    except Exception as e:
        print(f"Error initializing Gemini model: {str(e)}")
        model_gemini = None
else:
    model_gemini = None
    print("Warning: GOOGLE_API_KEY not found in environment variables. Poem generator will use fallback mode.")

def generate_acrostic_poem(name):
    # If no API key or model isn't initialized, use fallback immediately
    if not model_gemini:
        print("Using fallback mode: Gemini model not available")
        return generate_fallback_poem(name)

    # Create a prompt for Gemini
    prompt = f"""Create an acrostic poem about the tennis player {name}. 
    Each line should start with the corresponding letter of their name and relate to tennis.
    Make it personal to {name}'s playing style, achievements, or characteristics if they're a known player.
    If they're not a known player, create a general tennis-themed acrostic poem.
    Keep each line between 30-50 characters.
    Format: Just the poem lines, one per line, no letter prefixes."""
    
    try:
        # Call Gemini API
        response = model_gemini.generate_content(prompt)
        
        if not response or not response.text:
            print("Empty response from Gemini, using fallback")
            return generate_fallback_poem(name)
        
        # Get the generated poem lines
        poem_lines = response.text.strip().split('\n')
        
        # Format the poem with the name's letters
        name = name.upper()
        formatted_lines = []
        for letter, line in zip(name, poem_lines):
            if letter.isalpha():
                formatted_lines.append(f"{letter}: {line.strip()}")
        
        return "\n".join(formatted_lines)
    except Exception as e:
        print(f"Error generating poem with Gemini: {str(e)}")
        return generate_fallback_poem(name)

def generate_fallback_poem(name):
    """Fallback poem generation using predefined words"""
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
