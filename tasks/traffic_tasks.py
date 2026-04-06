def get_easy_env():
    from env.environment import TrafficEnv
    env = TrafficEnv(max_steps=50)
    return env


def get_medium_env():
    from env.environment import TrafficEnv
    env = TrafficEnv(max_steps=70)
    return env


def get_hard_env():
    from env.environment import TrafficEnv
    env = TrafficEnv(max_steps=100)
    return env