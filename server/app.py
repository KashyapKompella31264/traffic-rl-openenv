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


# ✅ REQUIRED ENTRYPOINT FUNCTION
def main():
    app.run(host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()