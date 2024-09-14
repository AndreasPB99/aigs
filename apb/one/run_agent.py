# %% Imports (add as needed) #############################################
import pickle

import gymnasium as gym  # not jax based
import jax
from jax import random, nn, numpy as jnp, grad, tree
from tqdm import tqdm
from collections import deque, namedtuple
from make_agent import env, entry,  your_policy_fn, loss_fn, convert_to_action, run_mlp

rng = random.PRNGKey(0)
run_iterations = 100
params = []
with open('agent.pickle', 'rb') as input_file:
    params = pickle.load(input_file)

obs, info = env.reset()
for i in tqdm(range(run_iterations)):
    rng, key = random.split(rng)
    action = your_policy_fn(env, key, params, obs, epsilon=100000)
    next_obs, reward, terminated, truncated, info = env.step(action)
    obs, info = next_obs, info if not (terminated | truncated) else env.reset()
