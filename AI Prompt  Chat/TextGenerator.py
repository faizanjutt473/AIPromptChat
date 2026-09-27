from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import threading
import json

load_dotenv()

app = Flask(__name__)
client = InferenceClient(api_key=os.getenv("HUG_API_KEY"))

# Global variables
stop_event = threading.Event()
saved_chats = {}  # Save chats with prompt name as key

@app.route('/')
def index():
    return render_template('index.html', saved_chats=saved_chats)

@app.route('/generate', methods=['POST'])
def generate():
    global stop_event
    user_input = request.json.get("message", "")
    prompt_name = request.json.get("prompt_name", "")  # Get prompt name for saving the chat

    if not user_input:
        return jsonify({"response": "Please enter a message."})

    messages = [{"role": "user", "content": user_input}]

    stop_event.clear()
    response_text = ""

    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            temperature=0.5,
            max_tokens=512,
            top_p=0.85,
            stream=True
        )

        for chunk in response:
            if stop_event.is_set():
                return jsonify({"response": "⚠️ Response generation stopped by user."})

            if chunk.choices and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content

        # Save the generated chat with prompt name
        if prompt_name:
            if prompt_name not in saved_chats:
                saved_chats[prompt_name] = []
            saved_chats[prompt_name].append({"user_input": user_input, "response": response_text})

        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"response": f"❌ Error: {str(e)}"})

@app.route('/stop', methods=['POST'])
def stop_generation():
    global stop_event
    stop_event.set()
    return jsonify({"status": "stopped"})

@app.route('/save_chat', methods=['POST'])
def save_chat():
    chat_data = request.json.get("chat", "")
    prompt_name = request.json.get("prompt_name", "")

    if chat_data and prompt_name:
        if prompt_name not in saved_chats:
            saved_chats[prompt_name] = []
        saved_chats[prompt_name].append(chat_data)

    return jsonify({"status": "saved", "saved_chats": saved_chats})

@app.route('/get_chats', methods=['GET'])
def get_chats():
    return jsonify({"saved_chats": saved_chats})



if __name__ == '__main__':
    app.run(debug=True)
