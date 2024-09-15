# %% Imports (add as needed) #############################################
import pickle

import gymnasium as gym  # not jax based
import jax
from jax import random, nn, numpy as jnp, grad, tree
from tqdm import tqdm
from collections import deque, namedtuple
from make_agent import entry,  your_policy_fn, loss_fn, convert_to_action, run_mlp, env as make_agent_env, run_episode as make_agent_run_episode, init_mlp

agent_file_name = 'agent'


def run():
    env = gym.make("CartPole-v1",  render_mode="human")
    rng = random.PRNGKey(0)
    run_iterations = 100
    with open(f'{agent_file_name}.pkl', 'rb') as input_file:
        params = pickle.load(input_file)

    obs, info = env.reset()
    for i in tqdm(range(run_iterations)):
        rng, key = random.split(rng)
        action = your_policy_fn(env, key, params, obs, exploration_rate=0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs, info = next_obs, info if not (terminated | truncated) else env.reset()


def train():
    rng = random.PRNGKey(0)
    obs, _ = make_agent_env.reset()
    params = init_mlp(len(obs), make_agent_env.action_space.__dict__['n'])
    make_agent_run_episode(make_agent_env, rng, obs, params)
    with open(f'{agent_file_name}.pkl', 'wb') as output_file:
        pickle.dump(params, output_file)

    make_agent_env.close()


if __name__ == '__main__':
    # train()
    run()
