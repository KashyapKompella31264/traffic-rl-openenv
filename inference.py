from agent.q_learning import QLearningAgent
from env.models import Action
from tasks.traffic_tasks import get_easy_env, get_medium_env, get_hard_env
from tasks.grader import grade_episode
import pickle
agent=QLearningAgent
with open("q_table.pkl", "rb") as f:
    agent.q = pickle.load(f)

def encode_state(obs):
    def bucket(x):
        if x < 3:
            return 0
        elif x < 6:
            return 1
        else:
            return 2

    return (
        bucket(obs.north),
        bucket(obs.south),
        bucket(obs.east),
        bucket(obs.west),
        obs.signal
    )


def run_task(name, env, agent):
    print(f"\n=== TASK: {name} ===")

    obs = env.reset()
    state = encode_state(obs)

    total_reward = 0
    done = False

    print("[START]")

    while not done:
        ns = obs.north + obs.south
        ew = obs.east + obs.west

        if abs(ns - ew) > 2:
            action_val = 0 if ns > ew else 1
        else:
            action_val = agent.choose_action(state)

        action = Action(signal=action_val)

        next_obs, reward, done, _ = env.step(action)
        next_state = encode_state(next_obs)

        print(f"[STEP] action={action_val} reward={round(reward,3)}")

        state = next_state
        obs = next_obs
        total_reward += reward

    print("[END]")

    score = grade_episode(total_reward, env.max_steps)

    print(f"{name} → Score: {score}")
    return score


if __name__ == "__main__":
    agent = QLearningAgent()

    # IMPORTANT: turn off exploration
    agent.epsilon = 0

    tasks = [
        ("Easy", get_easy_env()),
        ("Medium", get_medium_env()),
        ("Hard", get_hard_env()),
    ]

    results = {}

    for name, env in tasks:
        score = run_task(name, env, agent)
        results[name] = score

    print("\nFinal Results:")
    for k, v in results.items():
        print(f"{k}: {v}")