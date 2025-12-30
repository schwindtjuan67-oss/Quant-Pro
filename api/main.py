from fastapi import FastAPI
import uvicorn
import json
import os

app = FastAPI(title="QuantBot API", version="1.0")

STATE_FILE = "../Live/state_manager.json"  # Archivo generado por tu bot

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"status": "no state file found"}

@app.get("/")
def root():
    return {"message": "QuantBot API is running."}

@app.get("/state")
def get_state():
    return load_state()

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8081, reload=True)
