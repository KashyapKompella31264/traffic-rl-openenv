from flask import Flask, jsonify, request, send_from_directory
from env.environment import TrafficEnv
from env.models import Action
import os

app = Flask(__name__, static_folder="../static", static_url_path="/static")
env = TrafficEnv()


@app.route("/")
def home():
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), "..", "static"), "dashboard.html"
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/reset", methods=["GET", "POST"])
def reset():
    obs = env.reset()
    return jsonify(obs.dict())


@app.route("/step", methods=["POST"])
def step():
    data = request.get_json(force=True)
    signal = data.get("signal", 0)
    action = Action(signal=signal)
    obs, reward, done, info = env.step(action)
    return jsonify({
        "observation": obs.dict(),
        "reward": round(reward, 4),
        "done": done,
        "info": info,
    })


@app.route("/state", methods=["GET"])
def state():
    obs = env.state()
    return jsonify(obs.dict())


# ✅ REQUIRED ENTRYPOINT FUNCTION
def main():
    app.run(host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()