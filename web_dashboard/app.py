from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

STATE_FILE = "../Live/state_manager.json"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"status": "shadow not running"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/state")
def state_api():
    return jsonify(load_state())

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8090)
