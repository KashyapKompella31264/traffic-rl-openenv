from env.environment import TrafficEnv
from env.models import Action
from agent.q_learning import QLearningAgent
import pickle
env = TrafficEnv()
agent = QLearningAgent()
with open("q_table.pkl", "wb") as f:
    pickle.dump(agent.q, f)


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

for episode in range(800):
    obs = env.reset()
    state = encode_state(obs)

    done = False
    total_reward = 0

    while not done:
        action_val = agent.choose_action(state)  # must be 0 or 1
        action = Action(signal=action_val)

        next_obs, reward, done, _ = env.step(action)
        next_state = encode_state(next_obs)

        agent.update(state, action_val, reward, next_state)

        state = next_state
        total_reward += reward

    agent.epsilon = max(0.01, agent.epsilon * 0.995)

    avg_reward = total_reward / env.max_steps
    print(f"Episode {episode}, Avg Reward {round(avg_reward,2)}")