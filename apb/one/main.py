# %% Imports (add as needed) #############################################
import gymnasium as gym  # not jax based
import jax
from jax import random, nn, numpy as jnp, grad, tree
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
#memory = deque(maxlen=buffer_size)  # <- replay buffer
# define more as needed


# %% Model ###############################################################

def init_mlp(input_space, output_space, hidden_layers_1=12, hidden_layers_2=8):
    w1 = random.normal(rng, (input_space, hidden_layers_1)) * 0.01
    b1 = random.normal(rng, (hidden_layers_1,)) * 0.01

    w2 = random.normal(rng, (hidden_layers_1, hidden_layers_2)) * 0.01
    b2 = random.normal(rng, (hidden_layers_2,)) * 0.01

    w3 = random.normal(rng, (hidden_layers_2, output_space)) * 0.01
    b3 = random.normal(rng, (output_space,)) * 0.01
    params = [w1, b1, w2, b2, w3, b3]
    return params

def run_mlp(params, x_data):
    w1, b1, w2, b2, w3, b3 = params
    z = x_data @ w1 + b1
    z = nn.relu(z)  # <- activation
    z = z @ w2 + b2
    z = nn.relu(z)  # <- activation
    z = z @ w3 + b3
    z = nn.softmax(z)  # <- activation
    return z

def convert_to_action(input):
    best = jnp.argmax(input)
    return best.item()

@jax.jit
def loss_fn(params, curr_obs, next_obs, reward, action):
    gamma = 0.15
    return (reward + gamma * jnp.max(run_mlp(params, next_obs)) - run_mlp(params, curr_obs)[action])**2


def train_mlp(params, memory, grad_fn):
    # while memory:
    #     mem_entry = memory.popleft()
    #     grad_val = grad_fn(params, mem_entry.obs, mem_entry.next_obs, mem_entry.reward, mem_entry.action)
    #     params = tree.map(lambda p, g: p - 0.01 * g, params, grad_val)
    for i in memory:
        grad_val = grad_fn(params, i.obs, i.next_obs, i.reward, i.action)
        params = tree.map(lambda p, g: p - 0.01 * g, params, grad_val)
    return params

def random_policy_fn(env, rng): # action (shape: ())
    n = env.action_space.__dict__['n']
    return random.randint(rng, (1,), 0, n).item()

def your_policy_fn(env, rng, params, obs, epsilon=0.5):  # obs (shape: (2,)) to action (shape: ())
    rand_int = random.uniform(rng, shape=(1,))[0]
    if rand_int > epsilon:
        return random_policy_fn(env, rng)

    my_run = run_mlp(params, obs)
    action = convert_to_action(my_run)
    return action

def run_episode(env, rng):
    # %% Environment #########################################################
    l_memory = deque(maxlen=buffer_size)
    obs, info = env.reset()
    params = init_mlp(len(obs), env.action_space.__dict__['n'])
    batch_size = 32
    grad_fn = grad(loss_fn)
    for i in tqdm(range(training_range)):
        exploration_rate = 1 - (i / training_range) * experience_modifier
        rng, key = random.split(rng)
        action = your_policy_fn(env, key, params, obs, epsilon=exploration_rate)

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
            params = train_mlp(params, experiences, grad_fn)


run_episode(env, rng)
# for episode in range(episodes):
#     new_memory = run_episode(env, rng)
#     memory.appendleft(new_memory)

env.close()
