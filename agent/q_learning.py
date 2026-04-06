class QLearningAgent:
    def __init__(self):
        self.q = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, state):
        import random
        if random.random() < self.epsilon:
            return random.choice([0, 1])

        return max([0, 1], key=lambda a: self.get_q(state, a))

    def update(self, state, action, reward, next_state):
        max_next = max([self.get_q(next_state, a) for a in [0, 1]])

        self.q[(state, action)] = self.get_q(state, action) + self.alpha * (
            reward + self.gamma * max_next - self.get_q(state, action)
        )