import pandas as pd
from pyscript import document
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from textblob import TextBlob

# 1. Load Datasets
try:
    # Your original knowledge base
    kb_data = pd.read_csv("data.csv") 
    
    # Your renamed word database
    chat_data = pd.read_csv("unigram_freq.csv")
    
    # Mapping columns: unigram_freq usually uses 'word'
    # We'll treat the 'word' column as our training text
    train_text = chat_data['word'].astype(str).head(5000) # Using top 5k for speed
    
    # Creating dummy labels since frequency files don't have 'intents'
    # This allows the NN to initialize properly
    train_labels = ["vocabulary"] * len(train_text)
    
except Exception as e:
    print(f"File loading error: {e}")
    train_text = ['hello', 'hi', 'help', 'bye']
    train_labels = ['greeting', 'greeting', 'help', 'farewell']

# 2. Train the Neural Network
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_text)
y = train_labels

clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500)
clf.fit(X, y)

# 3. Response Logic
responses = {
    'vocabulary': "I recognize that word from my unigram database!",
    'greeting': "Hello! System online.",
    'help': "I am running a neural network on your word frequency data.",
    'unknown': "Input processed. That word exists in my neural pathways."
}

def process_message(event):
    input_text = document.querySelector("#user-input").value
    if not input_text: return

    # Sentiment Analysis
    analysis = TextBlob(input_text)
    polarity = analysis.sentiment.polarity
    
    mood_badge = document.querySelector("#mood-badge")
    if polarity > 0.1:
        mood, style, prefix = "Happy", "bg-emerald-500/20 text-emerald-400", "😊 [AI]: "
    elif polarity < -0.1:
        mood, style, prefix = "Angry", "bg-red-500/20 text-red-400", "😠 [AI]: "
    else:
        mood, style, prefix = "Neutral", "bg-slate-800 text-slate-400", "🤖 [AI]: "
    
    mood_badge.innerText = f"Mood: {mood}"
    mood_badge.className = f"px-4 py-1 rounded-full text-xs font-bold uppercase border border-white/10 {style}"

    # Prediction
    try:
        input_vec = vectorizer.transform([input_text.lower()])
        prediction = clf.predict(input_vec)[0]
        reply = responses.get(prediction, responses['unknown'])
    except:
        reply = "My neurons are reconfiguring. Try another word."

    append_to_chat("You", input_text, "text-blue-400")
    append_to_chat(prefix, reply, "text-slate-200")
    document.querySelector("#user-input").value = ""

def append_to_chat(sender, message, color):
    chat_box = document.querySelector("#chat-box")
    div = document.createElement("div")
    div.className = "max-w-3xl mx-auto space-y-1 mb-4"
    div.innerHTML = f"<div class='text-xs font-bold tracking-widest {color}'>{sender}</div>" \
                    f"<div class='bg-white/5 p-4 rounded-2xl border border-white/5'>{message}</div>"
    chat_box.appendChild(div)
    chat_box.scrollTop = chat_box.scrollHeight

document.querySelector("#loading").style.display = "none"
