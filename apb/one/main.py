# %% Imports (add as needed) #############################################
import gymnasium as gym  # not jax based
from jax import random
from tqdm import tqdm
from collections import deque, namedtuple

# %% Constants ###########################################################
env = gym.make("MountainCar-v0", render_mode="human")  #  render_mode="human")
rng = random.PRNGKey(0)
entry = namedtuple("Memory", ["obs", "action", "reward", "next_obs", "done"])

buffer_size = 1000
training_range = 200
episodes = 1
memory = deque(maxlen=buffer_size)  # <- replay buffer
# define more as needed

def init_mlp():
    hidden = 2
    w1 = random.normal(rng, (28 * 28, hidden)) * 0.01
    b1 = random.normal(rng, (hidden,)) * 0.01
    w2 = random.normal(rng, (hidden, 10)) * 0.01
    b2 = random.normal(rng, (10,)) * 0.01
    params = [w1, b1, w2, b2]
    return params

# %% Model ###############################################################
def random_policy_fn(env, rng, obs): # action (shape: ())
    n = env.action_space.__dict__['n']
    return random.randint(rng, (1,), 0, n).item()

def your_policy_fn(env, rng, obs):  # obs (shape: (2,)) to action (shape: ())
    random_int = random.randint(rng, (1,), 0, 11)
    if random_int == 0: # Random selection to offset
        return random_policy_fn(env, rng, obs)
    return random_policy_fn(env, rng, obs)


def train_mlp(mlp, memory):
    raise NotImplementedError

def run_episode(env, rng):
    # %% Environment #########################################################
    l_memory = deque(maxlen=buffer_size)
    obs, info = env.reset()
    for i in tqdm(range(training_range)):

        rng, key = random.split(rng)
        action = your_policy_fn(env, key, obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        l_memory.append(entry(obs, action, reward, next_obs, terminated | truncated))
        obs, info = next_obs, info if not (terminated | truncated) else env.reset()

        # If queue full enough, take random sample and train
    return l_memory


for episode in range(episodes):
    new_memory = run_episode(env, rng)
    memory.appendleft(new_memory)

env.close()
