
from flask import Flask
import subprocess

app = Flask(__name__)

@app.post("/start")
def start_streamlit():
    subprocess.Popen(["streamlit", "run", "rag3.py"])
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(port=5050)