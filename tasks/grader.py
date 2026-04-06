def grade_episode(total_reward, steps):
    """
    Normalize reward to 0–1
    """

    # worst case (all bad steps)
    worst = -steps

    # best case (all good steps)
    best = steps

    score = (total_reward - worst) / (best - worst)

    return round(max(0.0, min(1.0, score)), 4)