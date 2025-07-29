pip install transformers flask torch

from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
import re

app = Flask(__name__)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Initialize session log
LOG_FILE = "log.json"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as f:
        json.dump([], f)

# Basic offensive filter
def is_offensive(message):
    bad_words = ["hate", "stupid", "kill", "die", "worthless", "useless"]
    return any(word in message.lower() for word in bad_words)

@app.route('/')
def home():
    return render_template("chatbot.html")

@app.route('/get', methods=['POST'])
def get_bot_response():
    user_input = request.form['msg']

    if is_offensive(user_input):
        response = "I'm here to help, but please use respectful language."
    else:
        # Generate response
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([new_user_input_ids], dim=-1) if 'chat_history_ids' not in globals() else torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        global chat_history_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Log conversation
    with open(LOG_FILE, 'r+') as f:
        data = json.load(f)
        data.append({"user": user_input, "bot": response})
        f.seek(0)
        json.dump(data, f, indent=2)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
