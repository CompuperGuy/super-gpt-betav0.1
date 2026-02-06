import pandas as pd
from pyscript import document
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

# --- GLOBAL MODEL ASSETS ---
vectorizer = CountVectorizer()
clf = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=500)

def init_model():
    try:
        # Load unigram_freq.csv
        df = pd.read_csv("unigram_freq.csv")
        # Use top 2000 words as 'known' vocabulary for training
        train_text = df['word'].astype(str).head(2000).tolist()
        train_labels = ["recognized"] * len(train_text)
        
        # Add a few manual intents so it can actually "chat"
        train_text += ["hello", "hi", "hey", "bye", "goodbye", "who are you"]
        train_labels += ["greet", "greet", "greet", "exit", "exit", "info"]
        
        X = vectorizer.fit_transform(train_text)
        clf.fit(X, train_labels)
        
        document.querySelector("#status").innerText = "Online"
        document.querySelector("#status").className = "text-[10px] uppercase tracking-widest text-emerald-500"
    except Exception as e:
        print(f"Error: {e}")
        document.querySelector("#status").innerText = "Offline / Error"

# Responses Dictionary
RESPONSES = {
    "greet": "Hello! My neural network is active. How can I assist?",
    "exit": "System standby. Goodbye!",
    "info": "I am a browser-based AI running on Python and Scikit-Learn.",
    "recognized": "I recognize that word from my unigram database!",
    "unknown": "I processed that, but it's outside my current training data."
}

def process_message(event):
    user_input = document.querySelector("#user-input").value
    if not user_input: return

    # 1. Add User Message to UI
    append_message("You", user_input, "user-msg")
    document.querySelector("#user-input").value = ""

    # 2. Neural Prediction
    try:
        input_vec = vectorizer.transform([user_input.lower()])
        prediction = clf.predict(input_vec)[0]
        reply = RESPONSES.get(prediction, RESPONSES["unknown"])
    except:
        reply = "My neural engine is still initializing. One moment..."

    # 3. Add AI Message to UI
    append_message("Neural AI", reply, "ai-msg")

def append_message(sender, text, css_class):
    chat_box = document.querySelector("#chat-box")
    div = document.createElement("div")
    div.className = f"p-4 rounded-2xl max-w-[85%] text-sm shadow-md {css_class}"
    div.innerHTML = f"<b class='block text-[10px] uppercase mb-1 opacity-50'>{sender}</b>{text}"
    chat_box.appendChild(div)
    chat_box.scrollTop = chat_box.scrollHeight

# Run initialization
init_model()
