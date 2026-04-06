from flask import Flask, jsonify
from env.environment import TrafficEnv

app = Flask(__name__)
env = TrafficEnv()

@app.route("/reset", methods=["GET", "POST"])
def reset():
    obs = env.reset()
    return jsonify(obs.dict())

@app.route("/")
def home():
    return {"status": "ok"}

app = Flask(__name__)

# keep routes

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)