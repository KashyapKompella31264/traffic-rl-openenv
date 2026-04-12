"""
evaluate.py — Runs baseline comparisons across multiple strategies.
Outputs a formatted comparison table and per-strategy reward curves.
"""

from env.environment import TrafficEnv
from env.models import Action
from agent.q_learning import QLearningAgent
from tasks.grader import grade_episode
import pickle
import os
import json
import random


def encode_state(obs):
    def bucket(x):
        if x < 3:
            return 0
        elif x < 6:
            return 1
        else:
            return 2
    return (bucket(obs.north), bucket(obs.south), bucket(obs.east), bucket(obs.west), obs.signal)


def run_strategy(strategy_fn, env, num_episodes=20):
    """Run a strategy over multiple episodes and return aggregated results."""
    scores = []
    total_rewards = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            action_val = strategy_fn(obs)
            action = Action(signal=action_val)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            steps += 1

        score = grade_episode(ep_reward, steps)
        scores.append(score)
        total_rewards.append(ep_reward)

    return {
        "avg_score": round(sum(scores) / len(scores), 4),
        "max_score": round(max(scores), 4),
        "min_score": round(min(scores), 4),
        "avg_reward": round(sum(total_rewards) / len(total_rewards), 4),
        "scores": scores,
    }


# --- Strategies ---

def random_strategy(obs):
    """Randomly pick 0 or 1."""
    return random.choice([0, 1])


def fixed_ns_strategy(obs):
    """Always keep NS green."""
    return 0


def fixed_alternating_strategy(obs):
    """Alternate every step."""
    return 1 - obs.signal


def heuristic_strategy(obs):
    """Pick the direction with more cars."""
    ns = obs.north + obs.south
    ew = obs.east + obs.west
    if ns > ew:
        return 0
    elif ew > ns:
        return 1
    else:
        return obs.signal  # keep current if tied


def make_qtable_strategy(q_table):
    """Use Q-table to pick action."""
    def strategy(obs):
        state = encode_state(obs)

        q0 = q_table.get((state, 0), 0.0)
        q1 = q_table.get((state, 1), 0.0)
        return 0 if q0 >= q1 else 1

    return strategy


def make_hybrid_strategy(q_table):
    """Heuristic + Q-table fallback (the inference.py approach)."""
    def strategy(obs):
        ns = obs.north + obs.south
        ew = obs.east + obs.west
        if abs(ns - ew) > 2:
            return 0 if ns > ew else 1
        state = encode_state(obs)
        q0 = q_table.get((state, 0), 0.0)
        q1 = q_table.get((state, 1), 0.0)
        return 0 if q0 >= q1 else 1

    return strategy


if __name__ == "__main__":
    NUM_EPISODES = 30

    # Load Q-table
    q_table = {}
    if os.path.exists("q_table.pkl"):
        try:
            with open("q_table.pkl", "rb") as f:
                q_table = pickle.load(f)
            print(f"Loaded Q-table with {len(q_table)} entries\n")
        except Exception:
            print("⚠️  Could not load Q-table, using empty\n")

    strategies = {
        "Random":       random_strategy,
        "Fixed NS":     fixed_ns_strategy,
        "Alternating":  fixed_alternating_strategy,
        "Heuristic":    heuristic_strategy,
        "Q-Learning":   make_qtable_strategy(q_table),
        "Hybrid (H+Q)": make_hybrid_strategy(q_table),
    }

    env = TrafficEnv(max_steps=50)
    results = {}

    print("=" * 68)
    print(f"{'Strategy':<18} {'Avg Score':>10} {'Max Score':>10} {'Min Score':>10} {'Avg Reward':>12}")
    print("=" * 68)

    for name, fn in strategies.items():
        res = run_strategy(fn, env, num_episodes=NUM_EPISODES)
        results[name] = res
        print(f"{name:<18} {res['avg_score']:>10.4f} {res['max_score']:>10.4f} "
              f"{res['min_score']:>10.4f} {res['avg_reward']:>12.4f}")

    print("=" * 68)

    # Find best
    best = max(results, key=lambda k: results[k]["avg_score"])
    print(f"\n🏆 Best strategy: {best} (avg score: {results[best]['avg_score']:.4f})")

    # Save results
    save_results = {k: {sk: sv for sk, sv in v.items() if sk != "scores"} for k, v in results.items()}
    with open("evaluation_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print("✅ Results saved to evaluation_results.json")
