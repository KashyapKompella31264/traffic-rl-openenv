# Hackathon Optimization: Traffic RL OpenEnv

Optimize the project to maximize scores across all 5 evaluation parameters of the Meta PyTorch Hackathon. The current codebase works but is minimal—gaps exist in environment realism, spec compliance, testing, and documentation.

## User Review Required

> [!IMPORTANT]
> **Server framework change:** The OpenEnv spec mandates **FastAPI** with WebSocket support. The current Flask server will be replaced with FastAPI + uvicorn. This changes the server runtime but the environment logic stays the same.

> [!WARNING]
> **Breaking change to [step()](file:///d:/rl-openenv/env/environment.py#30-65) return:** Currently returns [(obs, reward, done, {})](file:///d:/rl-openenv/server/app.py#14-17). Will change to return a proper `StepResult` model with `observation`, `reward`, `done`, and `truncated` fields per OpenEnv spec. The `info` dict will be embedded in the observation model.

---

## Proposed Changes

### Environment Core ([env/](file:///d:/rl-openenv/tasks/traffic_tasks.py#13-17))

#### [MODIFY] [models.py](file:///d:/rl-openenv/env/models.py)
- Add richer [Observation](file:///d:/rl-openenv/env/models.py#4-10) with `step_count`, `total_waiting`, `avg_wait_per_direction`, `time_of_day`, `active_events` fields
- Add `StepResult` model (observation + reward + done + truncated)
- Add `EpisodeState` model for the [state()](file:///d:/rl-openenv/env/environment.py#66-68) endpoint
- Add field validators and constrained types (e.g., `signal: Literal[0, 1]`)

#### [MODIFY] [environment.py](file:///d:/rl-openenv/env/environment.py)
- **Realistic traffic simulation:**
  - Time-of-day based arrival rates (rush hour = higher density)
  - Per-direction configurable arrival rates
  - Stochastic car arrivals using Poisson distribution
  - Signal-switch delay penalty (yellow light simulation)
- **Dynamic events system:** Random events like accidents (block a lane), emergency vehicles (must yield), pedestrian crossing requests
- **Improved reward shaping:**
  - Throughput bonus: reward for cars cleared
  - Congestion penalty: exponential penalty at high queue lengths
  - Fairness component: penalize if one direction starves
  - Signal-switching cost: small penalty for toggling signal each step
- **Proper episode boundaries:** `terminated` (green-wave achieved) vs `truncated` (max_steps hit)
- **Rich `info` dict:** avg wait time, throughput, queue lengths, events active

#### [NEW] [__init__.py](file:///d:/rl-openenv/env/__init__.py)
- Export [TrafficEnv](file:///d:/rl-openenv/env/environment.py#5-68), [Observation](file:///d:/rl-openenv/env/models.py#4-10), [Action](file:///d:/rl-openenv/env/models.py#12-14), `StepResult`

---

### Tasks & Grading (`tasks/`)

#### [MODIFY] [traffic_tasks.py](file:///d:/rl-openenv/tasks/traffic_tasks.py)
- **Easy:** Low traffic, no events, 50 steps
- **Medium:** Moderate traffic, rush-hour patterns, pedestrian crossings, 75 steps
- **Hard:** High traffic, random accidents, emergency vehicles, asymmetric arrivals, 100 steps
- Each level returns a task config dict with scenario parameters, not just different `max_steps`

#### [MODIFY] [grader.py](file:///d:/rl-openenv/tasks/grader.py)
- Multi-metric grading:
  - **Throughput (40%):** Total cars cleared / max possible
  - **Fairness (25%):** Std dev of per-direction wait times (lower = better)
  - **Efficiency (20%):** Avg reward per step normalized
  - **Safety (15%):** Penalty for ignoring emergency vehicles or blocking pedestrians
- Difficulty-adjusted scoring with clear rubric
- Return detailed breakdown dict alongside final score

#### [NEW] [__init__.py](file:///d:/rl-openenv/tasks/__init__.py)

---

### Server (`server/`)

#### [MODIFY] [app.py](file:///d:/rl-openenv/server/app.py)
- **Migrate from Flask to FastAPI** with uvicorn
- Add all required endpoints:
  - `POST /reset` → returns [Observation](file:///d:/rl-openenv/env/models.py#4-10)
  - `POST /step` → accepts [Action](file:///d:/rl-openenv/env/models.py#12-14), returns `StepResult`
  - `GET /state` → returns `EpisodeState`
  - `GET /` → health check with environment metadata
  - `GET /tasks` → list available tasks
- Proper Pydantic request/response models
- Error handling with HTTP status codes

#### [NEW] [__init__.py](file:///d:/rl-openenv/server/__init__.py)

---

### Client (`client.py`)

#### [NEW] [client.py](file:///d:/rl-openenv/client.py)
- `TrafficEnvClient` class wrapping HTTP calls to the server
- Methods: [reset()](file:///d:/rl-openenv/server/app.py#8-12), [step(action)](file:///d:/rl-openenv/env/environment.py#30-65), [state()](file:///d:/rl-openenv/env/environment.py#66-68)
- Type-safe with Pydantic models
- Context manager support

---

### Agent (`agent/`)

#### [MODIFY] [q_learning.py](file:///d:/rl-openenv/agent/q_learning.py)
- Add type annotations
- Add docstrings
- Improve state bucketing for richer observation space

#### [NEW] [__init__.py](file:///d:/rl-openenv/agent/__init__.py)

---

### Training & Inference

#### [MODIFY] [train.py](file:///d:/rl-openenv/train.py)
- Proper training loop as a function (not module-level code)
- Save Q-table after training completes (not before)
- Print episode stats (reward, score, Q-table size)
- Support different difficulty configs

#### [MODIFY] [inference.py](file:///d:/rl-openenv/inference.py)
- Use task configs from [traffic_tasks.py](file:///d:/rl-openenv/tasks/traffic_tasks.py) instead of hardcoded [TrafficEnv()](file:///d:/rl-openenv/env/environment.py#5-68)
- Better error handling (remove bare `except:`)
- Use `StepResult` model

---

### Config & Build

#### [MODIFY] [openenv.yaml](file:///d:/rl-openenv/openenv.yaml)
- Add `tasks`, `grader`, `version`, `tags` fields
- Reference task configs and grader function

#### [MODIFY] [pyproject.toml](file:///d:/rl-openenv/pyproject.toml)
- Add `fastapi`, `uvicorn`, `httpx` dependencies
- Remove `flask`
- Add `pytest` as dev dependency
- Add `requires-python`

#### [MODIFY] [Dockerfile](file:///d:/rl-openenv/Dockerfile)
- Install from [pyproject.toml](file:///d:/rl-openenv/pyproject.toml) via pip
- Use uvicorn entrypoint
- Add health check
- Multi-stage or slim build

#### [DELETE] [requiements.txt](file:///d:/rl-openenv/requiements.txt)
- Typo in filename, dependencies already in [pyproject.toml](file:///d:/rl-openenv/pyproject.toml)

#### [NEW] [requirements.txt](file:///d:/rl-openenv/requirements.txt)
- Properly named, auto-generated from pyproject.toml deps

#### [NEW] [.dockerignore](file:///d:/rl-openenv/.dockerignore)
- Exclude `venv/`, `.git/`, `__pycache__/`, `*.pkl`

---

### Testing

#### [NEW] [tests/test_environment.py](file:///d:/rl-openenv/tests/test_environment.py)
- Test [reset()](file:///d:/rl-openenv/server/app.py#8-12) returns valid [Observation](file:///d:/rl-openenv/env/models.py#4-10)
- Test [step()](file:///d:/rl-openenv/env/environment.py#30-65) with valid actions returns `StepResult`
- Test episode terminates at `max_steps`
- Test reward bounds
- Test event system works

#### [NEW] [tests/test_grader.py](file:///d:/rl-openenv/tests/test_grader.py)
- Test score normalization to [0, 1]
- Test multi-metric breakdown
- Test edge cases (zero reward, max reward)

#### [NEW] [tests/test_models.py](file:///d:/rl-openenv/tests/test_models.py)
- Test Pydantic validation on all models
- Test serialization/deserialization

#### [NEW] [tests/test_server.py](file:///d:/rl-openenv/tests/test_server.py)
- Integration tests using FastAPI TestClient
- Test all endpoints return correct status codes and schemas

#### [NEW] [tests/__init__.py](file:///d:/rl-openenv/tests/__init__.py)

---

### Documentation

#### [MODIFY] [README.md](file:///d:/rl-openenv/README.md)
- Architecture diagram (Mermaid)
- Complete API reference for all endpoints
- Detailed evaluation criteria and scoring rubric
- Environment mechanics deep-dive (reward formula, events, traffic patterns)
- Installation and usage examples
- Fix typos

---

## Verification Plan

### Automated Tests
```bash
# Run all tests from project root
cd d:\rl-openenv
python -m pytest tests/ -v
```

### Server Smoke Test
```bash
# Start the server
cd d:\rl-openenv
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, test endpoints:
curl http://localhost:7860/
curl -X POST http://localhost:7860/reset
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"signal": 0}'
curl http://localhost:7860/state
```

### Docker Verification
```bash
cd d:\rl-openenv
docker build -t traffic-rl .
docker run -p 7860:7860 traffic-rl
# Then test endpoints as above
```

### Training Verification
```bash
cd d:\rl-openenv
python train.py
# Should print episode stats and save q_table.pkl
```
