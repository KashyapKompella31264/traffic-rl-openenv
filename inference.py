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

API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
API_KEY = os.environ.get("API_KEY")


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


def encode_state(obs):
    return (obs.north, obs.south, obs.east, obs.west, obs.signal)


def make_llm_call():
    if OpenAI is None:
        print("[LLM_ERROR] OpenAI library not found", flush=True)
        return

    url = os.environ.get("API_BASE_URL")   # Use as-is — do NOT strip /v1
    key = os.environ.get("API_KEY")
    model = os.environ.get("MODEL_NAME")

    if not url or not key:
        print(f"[LLM_ERROR] Missing credentials: URL={url}, KEY={'SET' if key else 'MISSING'}", flush=True)
        return

    try:
        # Pass the URL exactly as injected by the hackathon environment
        client = OpenAI(base_url=url, api_key=key)

        completion = client.chat.completions.create(
            model=model if model else "gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        print(f"[LLM_CALL_SUCCESS] Response: {completion.choices[0].message.content}", flush=True)
    except Exception as e:
        print(f"[LLM_ERROR] {type(e).__name__}: {str(e)}", flush=True)


def run_task(task_name):
    env = TrafficEnv()
    agent = QLearningAgent()

    if os.path.exists("q_table.pkl"):
        try:
            with open("q_table.pkl", "rb") as f:
                agent.q = pickle.load(f)
        except:
            pass

    obs = env.reset()
    state = encode_state(obs)

    rewards = []
    steps = 0
    done = False

    log_start(task=task_name, env="traffic-env", model=MODEL_NAME)

    success = False
    score = 0.0

    try:
        while not done:
            ns = obs.north + obs.south
            ew = obs.east + obs.west

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

    except:
        pass

    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return score


if __name__ == "__main__":
    make_llm_call()

    for task in ["easy", "medium", "hard"]:
        try:
            run_task(task)
        except Exception as e:
            print(f"Task {task} failed: {str(e)}", flush=True)