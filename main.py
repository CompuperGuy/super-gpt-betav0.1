import pandas as pd
import numpy as np
import random
import asyncio
from pyscript import document, window
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

# --- 1. THE GENERATIVE ENGINE (Markov Chain) ---
# Since we can't load 50GB of text, we build a "micro-language model"
# It learns which words follow which words.

class MicroLLM:
    def __init__(self):
        self.chain = {}
        self.starter_words = []
        
    def train(self, text_data):
        # Break text into words
        words = text_data.split()
        
        # Build pairs: "The cat" -> "sat"
        for i in range(len(words) - 2):
            current_pair = (words[i], words[i+1])
            next_word = words[i+2]
            
            if current_pair not in self.chain:
                self.chain[current_pair] = []
            self.chain[current_pair].append(next_word)
            
            # Keep track of words that start sentences
            if i == 0 or words[i-1].endswith('.'):
                self.starter_words.append(words[i])

    def generate(self, prompt_word, length=15):
        # Try to find a starting pair based on user input
        current_pair = None
        
        # Look for the user's word in our database
        matching_pairs = [k for k in self.chain.keys() if k[0].lower() == prompt_word.lower()]
        
        if matching_pairs:
            current_pair = random.choice(matching_pairs)
            output = [current_pair[0], current_pair[1]]
        else:
            # Fallback to random start
            if not self.chain: return "Model not trained yet."
            key = random.choice(list(self.chain.keys()))
            current_pair = key
            output = [key[0], key[1]]

        # Generate the rest of the sentence
        for _ in range(length):
            if current_pair in self.chain:
                next_word = random.choice(self.chain[current_pair])
                output.append(next_word)
                current_pair = (current_pair[1], next_word)
            else:
                break
                
        # Join and format
        sentence = " ".join(output)
        return sentence.capitalize() + "."

# --- 2. INITIALIZATION & TRAINING ---

# We need a text corpus to teach it how to write.
# Since unigram_freq.csv is just a word list, we will use a 
# built-in training corpus to give it grammar structure.
TRAINING_CORPUS = """
The neural network is a powerful tool for learning patterns. 
Artificial intelligence is changing the world of technology rapidly.
Data science involves analyzing large datasets to find trends.
Machine learning models can predict outcomes with high accuracy.
The future of computing lies in quantum mechanics and biological interfaces.
Taurus is a generative model running in the browser.
Python is a great language for data analysis and scripting.
The quick brown fox jumps over the lazy dog repeatedly.
Deep learning uses layers of nodes to simulate the human brain.
I am a chatbot designed to assist you with your queries.
Web assembly allows python to run inside chrome and firefox.
"""

llm = MicroLLM()
llm.train(TRAINING_CORPUS)

# --- 3. CLASSIFIER (For the Graph Visualization) ---
# We keep the MLP to create the "Training Graph" effect
vectorizer = CountVectorizer()
clf = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1, warm_start=True)
dummy_X = vectorizer.fit_transform(["hello", "test", "data", "ai"])
dummy_y = [0, 1, 0, 1]

# --- 4. FUNCTIONS ---

def append_chat(sender, text, style):
    box = document.querySelector("#chat-box")
    div = document.createElement("div")
    div.className = f"message {style}"
    div.innerHTML = f"<b>{sender}:</b> {text}"
    box.appendChild(div)
    box.scrollTop = box.scrollHeight

async def process_input(event):
    user_input = document.querySelector("#user-input").value
    if not user_input: return
    
    append_chat("You", user_input, "msg-user")
    document.querySelector("#user-input").value = ""
    
    # Simulate "Thinking" / Training Loop
    loss = 1.0
    for i in range(5):
        # Fake a training step to update the graph
        loss = loss * 0.8 + (random.random() * 0.1)
        window.updateGraph(i + 1, loss)
        await asyncio.sleep(0.1) # Updates UI without freezing
        
    # Generate Text
    # 1. Grab the last word the user said
    last_word = user_input.split()[-1]
    
    # 2. Generate a sentence starting with that word
    response = llm.generate(last_word)
    
    append_chat("Taurus", response, "msg-ai")

# Initial Message
append_chat("System", "Training complete. Generative Engine Ready.", "msg-ai")
