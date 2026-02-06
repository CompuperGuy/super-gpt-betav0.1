import pandas as pd
from pyscript import document
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

# Initializing Global Vars
vectorizer = CountVectorizer()
clf = MLPClassifier(hidden_layer_sizes=(12, 12), max_iter=500)

def init_system():
    try:
        # Loading your unigram word frequency file
        df = pd.read_csv("unigram_freq.csv")
        # Training on top 3000 words for the v0.1 test
        train_text = df['word'].astype(str).head(3000).tolist()
        labels = ["DATA_NODE"] * len(train_text)
        
        # Hardcoding core Taurus responses
        train_text += ["hello", "status", "who are you", "exit"]
        labels += ["GREET", "STATUS", "IDENTITY", "EXIT"]
        
        X = vectorizer.fit_transform(train_text)
        clf.fit(X, labels)
        
        document.querySelector("#status").innerText = "SYSTEM READY: TAURUS v0.1 CORE ONLINE"
    except Exception as e:
        document.querySelector("#status").innerText = f"CRITICAL ERROR: {str(e)}"

RESPONSES = {
    "GREET": "TAURUS INTERFACE ACTIVE. HELLO USER.",
    "STATUS": "NEURAL NETWORK STABLE. 12 LAYERS DETECTED.",
    "IDENTITY": "I AM TAURUS v0.1. A NEURAL CLASSIFIER RUNNING IN WEB-ASSEMBLY.",
    "DATA_NODE": "INPUT RECOGNIZED IN LOCAL WORD DATABASE.",
    "EXIT": "SHUTTING DOWN... JOKE. I HAVE NO POWER TO QUIT.",
    "UNKNOWN": "COMMAND NOT FOUND IN TAURUS ARCHIVE."
}

def process_message(event):
    msg = document.querySelector("#user-input").value
    if not msg: return
    
    # User Line
    append_log(f"USER: {msg}", "black")
    document.querySelector("#user-input").value = ""

    # Prediction Logic
    try:
        v = vectorizer.transform([msg.lower()])
        p = clf.predict(v)[0]
        reply = RESPONSES.get(p, RESPONSES["UNKNOWN"])
    except:
        reply = "WAITING FOR INITIALIZATION..."

    append_log(f"TAURUS: {reply}", "red")

def append_log(text, color):
    box = document.querySelector("#chat-box")
    div = document.createElement("div")
    div.style.color = color
    div.innerText = text
    box.appendChild(div)
    box.scrollTop = box.scrollHeight

init_system()
