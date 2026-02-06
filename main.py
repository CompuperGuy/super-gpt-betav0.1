import pandas as pd
from pyscript import document
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from textblob import TextBlob

# 1. Load Datasets
# 'data.csv' is your large knowledge base
# 'data2.csv' should have columns: 'text' and 'intent'
try:
    kb_data = pd.read_csv("data.csv")
    chat_data = pd.read_csv("data2.csv")
except:
    # Fallback if files aren't found during testing
    chat_data = pd.DataFrame({
        'text': ['hello', 'bye', 'who are you', 'help'],
        'intent': ['greeting', 'farewell', 'identity', 'help']
    })

# 2. Train the Neural Network (Multi-Layer Perceptron)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(chat_data['text'])
y = chat_data['intent']

# Hidden layers (10, 10) creates a simple deep neural network architecture
clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
clf.fit(X, y)

# 3. Response Database
responses = {
    'greeting': "Greetings! My neural pathways are primed and ready.",
    'farewell': "Logging off. System standby...",
    'identity': "I am a Neural Network trained on your custom CSV data.",
    'help': "I can analyze sentiment and classify intents from your database.",
    'unknown': "Input processed, but intent is outside current training parameters."
}

def process_message(event):
    input_text = document.querySelector("#user-input").value
    if not input_text: return

    # A. Sentiment Analysis (The "Mood" Bit)
    analysis = TextBlob(input_text)
    polarity = analysis.sentiment.polarity
    
    mood_badge = document.querySelector("#mood-badge")
    if polarity > 0.1:
        mood = "Happy / Friendly"
        mood_badge.className = "px-4 py-1 rounded-full bg-emerald-500/20 text-emerald-400 text-xs font-bold uppercase border border-emerald-500/50"
        ai_prefix = "😊 [Happy AI]: "
    elif polarity < -0.1:
        mood = "Angry / Critical"
        mood_badge.className = "px-4 py-1 rounded-full bg-red-500/20 text-red-400 text-xs font-bold uppercase border border-red-500/50"
        ai_prefix = "😠 [Angry AI]: "
    else:
        mood = "Neutral"
        mood_badge.className = "px-4 py-1 rounded-full bg-slate-800 text-xs font-bold uppercase border border-white/10"
        ai_prefix = "🤖 [AI]: "
    
    mood_badge.innerText = f"Mood: {mood}"

    # B. Neural Network Prediction
    input_vec = vectorizer.transform([input_text.lower()])
    intent = clf.predict(input_vec)[0]
    reply = responses.get(intent, responses['unknown'])

    # C. Update UI
    append_to_chat("You", input_text, "text-blue-400")
    append_to_chat(ai_prefix, reply, "text-slate-200")
    
    document.querySelector("#user-input").value = ""

def append_to_chat(sender, message, color):
    chat_box = document.querySelector("#chat-box")
    div = document.createElement("div")
    div.className = "max-w-3xl mx-auto space-y-1"
    div.innerHTML = f"<div class='text-xs font-bold uppercase tracking-widest {color}'>{sender}</div>" \
                    f"<div class='bg-white/5 p-4 rounded-2xl border border-white/5'>{message}</div>"
    chat_box.appendChild(div)
    div.scrollIntoView()

# Hide loading screen
document.querySelector("#loading").style.display = "none"
