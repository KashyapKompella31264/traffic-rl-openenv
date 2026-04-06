# 🚦 Traffic RL OpenEnv Environment

## 📌 Overview

This project implements a **Reinforcement Learning (RL) based traffic signal control system** using the OpenEnv framework.

The agent learns to **optimize traffic flow** by dynamically switching signals based on real-time vehicle density at an intersection.

---

## 🎯 Problem Statement

Traditional traffic signals follow fixed timers, which often lead to congestion.
This project models traffic control as a **sequential decision-making problem**, where an RL agent minimizes total waiting time.

---

## 🧠 Approach

- Environment: Custom OpenEnv-compatible traffic simulation
- Agent: Q-Learning based policy with heuristic enhancement
- Objective: Minimize congestion (total waiting vehicles)

### State Representation

```
(north, south, east, west, signal)
```

### Actions

```
0 → North-South Green
1 → East-West Green
```

### Reward Function

- Positive reward → traffic reduced
- Negative reward → congestion increased
- Clipped to range: **[-1, +1]**

---

## ⚙️ Features

- ✅ OpenEnv compliant environment (`reset`, `step`, `state`)
- ✅ Multiple task configurations (easy, medium, hard)
- ✅ Normalized grading system (0.0 – 1.0)
- ✅ Structured inference logs (`[START] [STEP] [END]`)
- ✅ Dockerized deployment
- ✅ Hugging Face Spaces integration

---

## 📊 Evaluation

The system is evaluated using a grader that normalizes total reward:

```
Score ∈ [0.0, 1.0]
```

Higher score → better traffic optimization

---

## 🚀 Deployment

### Hugging Face Space

- UI: https://huggingface.co/spaces/kashyapkompella/traffic-rl-openenv
- API: https://kashyapkompella-traffic-rl-openenv.hf.space

### Endpoints

```
POST /reset → initialize environment
GET  /      → health check
```

---

## 🐳 Docker

Build and run:

```bash
docker build -t traffic-rl .
docker run -p 7860:7860 traffic-rl
```

---

## 📂 Project Structure

```
env/        → environment logic
agent/      → RL agent
tasks/      → task configurations & grading
server/     → API server
inference.py → evaluation script
```

---

## 🔮 Future Improvements

- Deep RL (DQN / PPO)
- Multi-intersection coordination
- Real-time traffic data integration
- Visualization dashboard

---

## 👤 Author

Kashyap

---

## 🏁 Summary

This project demonstrates a complete pipeline:
**RL model → Environment → Evaluation → Deployment**

Designed to showcase practical application of RL in real-world systems like traffic optimization.
