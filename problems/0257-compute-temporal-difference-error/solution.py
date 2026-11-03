import numpy as np

def compute_td_error(v_s: float, reward: float, v_s_prime: float, gamma: float, done: bool) -> float:
    if done:
        td_target = reward
    else:
        td_target = reward + gamma * v_s_prime

    td_error = td_target - v_s
    return td_error

    