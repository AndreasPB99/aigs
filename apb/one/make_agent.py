# %% Imports (add as needed) #############################################
import pickle

import gymnasium as gym  # not jax based
import jax
from jax import random, nn, numpy as jnp, grad, tree, jit
from tqdm import tqdm
from collections import deque, namedtuple

# %% Constants ###########################################################
env = gym.make("CartPole-v1", render_mode="human")  #  render_mode="human")
rng = random.PRNGKey(0)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])

buffer_size = 1000
training_range = 2000
episodes = 1
experience_modifier = 1.5


def init_mlp(input_space, output_space, hidden_layers_1=8, hidden_layers_2=4):
    """
    Initialises the mlp, hardcoded with hidden layers.
    Output will be a list of the weights and biases
    """
    w1 = random.normal(rng, (input_space, hidden_layers_1)) * 0.01
    b1 = random.normal(rng, (hidden_layers_1,)) * 0.01

    w2 = random.normal(rng, (hidden_layers_1, hidden_layers_2)) * 0.01
    b2 = random.normal(rng, (hidden_layers_2,)) * 0.01

    w3 = random.normal(rng, (hidden_layers_2, output_space)) * 0.01
    b3 = random.normal(rng, (output_space,)) * 0.01
    params = [w1, b1, w2, b2, w3, b3]
    return params


def run_mlp(params, input):
    """
    Runs the input through the mlp (here named params) and returns the output
    """
    w1, b1, w2, b2, w3, b3 = params
    z = input @ w1 + b1
    z = nn.relu(z)  # <- activation
    z = z @ w2 + b2
    z = nn.relu(z)  # <- activation
    z = z @ w3 + b3
    z = nn.softmax(z)  # <- activation
    return z


def convert_to_action(input):
    """
    Gets the action from the output of the mlp
    """
    best = jnp.argmax(input)
    return best.item()


@jax.jit
def loss_fn(params, curr_obs, next_obs, reward, action):
    """
        Standard loss function for deep reinforcement learning
    """
    return (reward + 0.1 * jnp.max(run_mlp(params, next_obs)) - run_mlp(params, curr_obs)[action])**2

# Creates a just in time compiled version of the differentiated loss function
grad_loss_fn = jit(grad(loss_fn))


def train_mlp(params, memory):
    """
    Trains the mlp on the experiences stored in memory and updates the params based on that
    """
    for i in memory:
        grad_val = grad_loss_fn(params, i.obs, i.next_obs, i.reward, i.action)
        params = tree.map(lambda p, g: p - 0.01 * g, params, grad_val)
    return params


def random_policy_fn(env, rng): # action (shape: ())
    """
    A policy that picks a random action
    """
    n = env.action_space.__dict__['n']
    return random.randint(rng, (1,), 0, n).item()


def your_policy_fn(env, rng, params, obs, exploration_rate=0.5):  # obs (shape: (2,)) to action (shape: ())
    """
    Policy that picks a random action more frequently at a low epsilon and used the mlp whenever it is not exploring
    """
    rand_int = random.uniform(rng, shape=(1,))[0]
    if rand_int < exploration_rate:
        return random_policy_fn(env, rng)

    my_run = run_mlp(params, obs)
    action = convert_to_action(my_run)
    return action


def run_episode(env, rng, obs, params):
    """
    Runs the current episode with params
    """
    l_memory = deque(maxlen=buffer_size)
    batch_size = 32
    for i in tqdm(range(training_range)):
        exploration_rate = 1 - (i / training_range) * experience_modifier
        rng, key = random.split(rng)
        action = your_policy_fn(env, key, params, obs, exploration_rate=exploration_rate)

        next_obs, reward, terminated, truncated, info = env.step(action)

        if terminated: # makes reward negative if the game is lost
            reward -= 10
        elif truncated:
            reward += 10

        l_memory.append(entry(obs, action, reward, next_obs, terminated | truncated))
        obs, info = next_obs, info if not (terminated | truncated) else env.reset()

        # If queue full enough, take random sample and train
        if len(l_memory) > batch_size:
            idx = random.choice(key, len(l_memory), (batch_size,), replace=True)
            experiences = [l_memory[i] for i in idx]
            params = train_mlp(params, experiences)


if __name__ == '__main__':
    obs, _ = env.reset()
    params = init_mlp(len(obs), env.action_space.__dict__['n'])
    run_episode(env, rng, obs, params)
    with open('agent.pickle', 'wb') as output_file:
        pickle.dump(params, output_file)

    env.close()
