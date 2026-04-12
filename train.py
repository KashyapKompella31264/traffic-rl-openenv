from env.environment import TrafficEnv
from env.models import Action
from agent.q_learning import QLearningAgent
import pickle
import json

env = TrafficEnv()
agent = QLearningAgent()


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


NUM_EPISODES = 800
training_log = []

for episode in range(NUM_EPISODES):
    obs = env.reset()
    state = encode_state(obs)

    done = False
    total_reward = 0
    steps = 0

    while not done:
        action_val = agent.choose_action(state)
        action = Action(signal=action_val)

        next_obs, reward, done, _ = env.step(action)
        next_state = encode_state(next_obs)

        agent.update(state, action_val, reward, next_state)

        state = next_state
        total_reward += reward
        steps += 1

    agent.epsilon = max(0.01, agent.epsilon * 0.995)

    avg_reward = round(total_reward / env.max_steps, 4)
    training_log.append({
        "episode": episode,
        "avg_reward": avg_reward,
        "epsilon": round(agent.epsilon, 4),
        "total_reward": round(total_reward, 4),
        "q_table_size": len(agent.q)
    })

    if episode % 50 == 0 or episode == NUM_EPISODES - 1:
        print(f"Episode {episode:4d} | Avg Reward: {avg_reward:+.4f} | "
              f"Epsilon: {agent.epsilon:.4f} | Q-table size: {len(agent.q)}")

# Save trained Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(agent.q, f)
print(f"\n✅ Q-table saved ({len(agent.q)} entries)")

# Save training metrics
with open("training_log.json", "w") as f:
    json.dump(training_log, f, indent=2)
print(f"✅ Training log saved ({len(training_log)} episodes)")