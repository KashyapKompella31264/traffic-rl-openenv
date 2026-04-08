import os
import pickle
from typing import List, Optional
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from env.environment import TrafficEnv
from env.models import Action
from agent.q_learning import QLearningAgent
from tasks.grader import grade_episode


# ===== ENV VARIABLES (MANDATORY) =====
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "dummy-model")
HF_TOKEN = os.getenv("HF_TOKEN")

client = None
if OpenAI and HF_TOKEN:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ===== LOGGING FUNCTIONS (STRICT FORMAT) =====
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ===== STATE ENCODING =====
def encode_state(obs):
    return (obs.north, obs.south, obs.east, obs.west, obs.signal)


# ===== RUN TASK =====
def run_task(task_name):
    env = TrafficEnv()
    agent = QLearningAgent()

    # load trained Q-table
    with open("q_table.pkl", "rb") as f:
        agent.q = pickle.load(f)

    obs = env.reset()
    state = encode_state(obs)

    rewards = []
    steps = 0
    done = False

    log_start(task=task_name, env="traffic-env", model=MODEL_NAME)

    try:
        while not done:
            ns = obs.north + obs.south
            ew = obs.east + obs.west

            # hybrid policy
            if abs(ns - ew) > 2:
                action_val = 0 if ns > ew else 1
            else:
                action_val = agent.choose_action(state)

            action = Action(signal=action_val)

            next_obs, reward, done, _ = env.step(action)

            rewards.append(reward)
            steps += 1

            log_step(
                step=steps,
                action=str(action_val),
                reward=reward,
                done=done,
                error=None
            )

            state = encode_state(next_obs)
            obs = next_obs

        score = grade_episode(sum(rewards), steps)
        success = score > 0.3

    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return score


if __name__ == "__main__":
    try:
        run_task("easy")
        run_task("medium")
        run_task("hard")
    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)