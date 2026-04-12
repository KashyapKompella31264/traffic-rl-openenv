---
title: Traffic Rl Openenv
emoji: 🚦
colorFrom: green
colorTo: red
sdk: docker
pinned: false
short_description: RL-based Traffic Signal Control with Live Dashboard
---

# 🚦 Traffic RL OpenEnv

> **Reinforcement Learning Traffic Signal Controller** — An AI agent that learns to dynamically control traffic lights at a 4-way intersection, minimizing vehicle wait times through Q-Learning with heuristic enhancement.

---

## 📌 Problem Statement

Traditional traffic signals follow **fixed timers**, causing unnecessary congestion during uneven traffic loads. This project models traffic control as a **sequential decision-making problem**, where an RL agent observes real-time vehicle density and dynamically switches signals to **minimize total waiting vehicles**.

---

## 🧠 Approach

| Component | Details |
|-----------|---------|
| **Environment** | Custom OpenEnv-compatible 4-way intersection simulation |
| **Agent** | Q-Learning (tabular) with ε-greedy exploration |
| **Enhancement** | Heuristic fallback + LLM integration for inference |
| **Objective** | Minimize congestion across all directions |

### State Space
```
(bucket(north), bucket(south), bucket(east), bucket(west), signal)
```
Where `bucket(x)` maps: `0-2 → 0`, `3-5 → 1`, `6+ → 2`

### Action Space
| Action | Effect |
|--------|--------|
| `0` | North-South Green |
| `1` | East-West Green |

### Reward Design
- **Positive** → total waiting vehicles decreased
- **Negative** → congestion increased
- **Bonus** → small reward for serving the green direction
- **Clamped** to `[-1, +1]`

---

## ⚙️ Features

- ✅ OpenEnv compliant environment (`reset`, `step`, `state`)
- ✅ **Interactive Live Dashboard** with real-time intersection visualization
- ✅ Multiple play modes: Manual, Heuristic, Q-Table
- ✅ **6-strategy baseline comparison** (Random, Fixed, Alternating, Heuristic, Q-Learning, Hybrid)
- ✅ Normalized grading system (0.0 – 1.0)
- ✅ Structured inference logs (`[START] [STEP] [END]`)
- ✅ LLM-enhanced inference with fallback chain
- ✅ Dockerized deployment with training-at-build
- ✅ Hugging Face Spaces integration

---

## 📊 Evaluation

The system is evaluated using multiple strategies over 30 episodes each:

```
Strategy           Avg Score   Max Score   Min Score   Avg Reward
====================================================================
Random               ~0.50      ~0.58       ~0.42       ~0.00
Fixed NS             ~0.50      ~0.56       ~0.44       ~0.00
Alternating          ~0.50      ~0.57       ~0.43       ~0.00
Heuristic            ~0.54      ~0.62       ~0.46       ~0.40
Q-Learning           ~0.56      ~0.64       ~0.48       ~0.60
Hybrid (H+Q)         ~0.57      ~0.65       ~0.49       ~0.70
====================================================================
🏆 Best: Hybrid (Heuristic + Q-Learning)
```

> Run `python evaluate.py` to generate fresh comparison results.

---

## 🖥️ Live Dashboard

The project includes an **interactive real-time dashboard** accessible at the root URL:

- 🟢 Animated intersection with directional car counts
- 📈 Live reward chart tracking agent performance
- 🎮 Three play modes: Manual, Heuristic, Q-Table
- ⚡ Auto-play mode for hands-free evaluation
- 📋 Structured inference log

---

## 🚀 Deployment

### Hugging Face Spaces
- **UI**: https://huggingface.co/spaces/kashyapkompella/traffic-rl-openenv
- **API**: https://kashyapkompella-traffic-rl-openenv.hf.space

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Live Dashboard |
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment → returns observation |
| `/step` | POST | Take action `{"signal": 0}` → returns obs, reward, done |
| `/state` | GET | Get current observation |

---

## 🐳 Docker

```bash
docker build -t traffic-rl .
docker run -p 7860:7860 traffic-rl
```

The Docker build **trains the Q-Learning agent** during image build, so the deployed container always has a populated model.

---

## 📂 Project Structure

```
env/
├── models.py          # Observation & Action Pydantic schemas
└── environment.py     # TrafficEnv simulation (reset, step, state)

agent/
└── q_learning.py      # Tabular Q-Learning agent (ε-greedy)

tasks/
├── traffic_tasks.py   # Difficulty configs (easy/medium/hard)
└── grader.py          # Episode grading (normalize to 0–1)

server/
└── app.py             # Flask API + dashboard server

static/
└── dashboard.html     # Interactive visualization dashboard

train.py               # Training script (800 episodes)
inference.py           # Multi-strategy inference (LLM + heuristic + Q-table)
evaluate.py            # Baseline comparison across 6 strategies
q_table.pkl            # Trained Q-table (serialized)
openenv.yaml           # OpenEnv framework manifest
Dockerfile             # Containerized deployment
```

---

## 🔮 Future Improvements

- Deep RL (DQN / PPO) with neural network function approximation
- Multi-intersection coordination
- Real-time traffic data integration
- Extended state space (pedestrians, emergency vehicles)

---

## 👤 Author

**Kashyap**

---

## 🏁 Summary

```
RL Training → Q-Table → Heuristic Enhancement → Live Dashboard → Evaluation → Deployment
```

A complete end-to-end pipeline demonstrating practical RL application in real-world traffic optimization, deployed on Hugging Face Spaces with an interactive visualization dashboard.
