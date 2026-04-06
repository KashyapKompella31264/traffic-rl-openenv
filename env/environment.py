from env.models import Observation, Action
import random


class TrafficEnv:
    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.cars = {
            "N": random.randint(0, 5),
            "S": random.randint(0, 5),
            "E": random.randint(0, 5),
            "W": random.randint(0, 5),
        }
        self.signal = 0  # 0 = NS green, 1 = EW green
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return Observation(
            north=self.cars["N"],
            south=self.cars["S"],
            east=self.cars["E"],
            west=self.cars["W"],
            signal=self.signal
        )

    def step(self, action: Action):
        prev_total_wait = sum(self.cars.values())

        # apply action
        self.signal = action.signal

        # cars passing
        if self.signal == 0:  # NS green
            self.cars["N"] = max(0, self.cars["N"] - 2)
            self.cars["S"] = max(0, self.cars["S"] - 2)
        else:  # EW green
            self.cars["E"] = max(0, self.cars["E"] - 2)
            self.cars["W"] = max(0, self.cars["W"] - 2)

        # new cars arrive
        for d in self.cars:
            self.cars[d] += random.choice([0, 1, 0])

        self.steps += 1

        # compute reward
        current_total_wait = sum(self.cars.values())

        max_wait = 40  # normalization constant
        reward = (prev_total_wait - current_total_wait) / 10
        if self.signal == 0:
            reward += (self.cars["N"] + self.cars["S"]) * 0.01
        else:
            reward += (self.cars["E"] + self.cars["W"]) * 0.01
        # clamp reward (safety)
        reward = max(-1.0, min(1.0, reward))

        done = self.steps >= self.max_steps

        return self._get_obs(), reward, done, {}
    
    def state(self):
        return self._get_obs()