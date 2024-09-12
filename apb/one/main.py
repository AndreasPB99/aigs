# %% Imports (add as needed) #############################################
import gymnasium as gym  # not jax based
from jax import random, nn
from tqdm import tqdm
from collections import deque, namedtuple

# %% Constants ###########################################################
env = gym.make("CartPole-v1", render_mode="human")  #  render_mode="human")
rng = random.PRNGKey(0)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])

buffer_size = 1000
training_range = 200
episodes = 1
#memory = deque(maxlen=buffer_size)  # <- replay buffer
# define more as needed


# %% Model ###############################################################

def init_mlp(input_space, output_space, hidden_layers=4):
    w1 = random.normal(rng, (input_space, hidden_layers)) * 0.01
    b1 = random.normal(rng, (hidden_layers,)) * 0.01
    w2 = random.normal(rng, (hidden_layers, output_space)) * 0.01
    b2 = random.normal(rng, (output_space,)) * 0.01
    params = [w1, b1, w2, b2]
    return params

def model(params, x_data):
    w1, b1, w2, b2 = params
    z = x_data @ w1 + b1
    z = nn.relu(z)  # <- activation
    z = z @ w2 + b2
    z = nn.softmax(z)  # <- activation
    return z

def convert_to_action(input):
    return int(round(max(input)))

def random_policy_fn(env, rng, obs): # action (shape: ())
    n = env.action_space.__dict__['n']
    return random.randint(rng, (1,), 0, n).item()

def your_policy_fn(env, rng, obs):  # obs (shape: (2,)) to action (shape: ())
    random_int = random.randint(rng, (1,), 0, 11)
    if random_int == 0: # Random selection to offset
        return random_policy_fn(env, rng, obs)
    return random_policy_fn(env, rng, obs)


def train_mlp(params, memory):
    while memory:
        value = memory.popleft()
    return params

def run_episode(env, rng):
    # %% Environment #########################################################
    l_memory = deque(maxlen=buffer_size)
    obs, info = env.reset()
    params = init_mlp(len(obs), env.action_space.__dict__['n'])
    batch_size = 32
    for i in tqdm(range(training_range)):
        rng, key = random.split(rng)
        action = your_policy_fn(env, key, obs)

        test = model(params, obs)
        my_action = convert_to_action(test)

        next_obs, reward, terminated, truncated, info = env.step(my_action)
        l_memory.append(entry(obs, action, reward, next_obs, terminated | truncated))
        obs, info = next_obs, info if not (terminated | truncated) else env.reset()

        # If queue full enough, take random sample and train
        if len(l_memory) > batch_size:
            params = train_mlp(params, l_memory)

run_episode(env, rng)
# for episode in range(episodes):
#     new_memory = run_episode(env, rng)
#     memory.appendleft(new_memory)

env.close()
