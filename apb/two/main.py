import jax
import gym
import gym_sokoban
import jax.numpy as jnp
from jax import random, grad, jit

rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
rngs = jax.random.split(_rng, 3)

# Create environment
#env = gym.make("CarRacing-v2")
env = gym.make("Sokoban-v1")
env.reset()
env.render('human')
obss = [env.step(env.action_space.sample())[0] for _ in range(1000)]
data = jnp.array(obss)

