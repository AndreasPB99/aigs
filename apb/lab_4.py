# %% lab_4.py
#   evolve neural networks (no grad) with ES/GA
# by: Noah Syrkis
# %% Imports ############################################################
from jax import random, grad, nn
from jaxtyping import Array
import jax.numpy as jnp
import evosax  # <- use this (https://github.com/RobertTLange/evosax)
import evojax  # <- or this  (https://github.com/google/evojax)
import plotly.graph_objects as go
import os
import plotly.express as px
import plotly.offline as pyo
from typing import Callable


rng, key = random.split(random.PRNGKey(0))

# %% Helper functions ###################################################
def plot_fn(fn: Callable, steps=100, radius = 4) -> None:   # plot a 3D function

    # create a grid of x and y values
    x = jnp.linspace(-radius, radius, steps)  # <- create a grid of x values
    y = jnp.linspace(-radius, radius, steps)  # <- create a grid of y values
    Z = fn(*jnp.meshgrid(x, y))  # <- evaluate the function on the grid

    # create a 3D surface plot
    fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])  # <- create a 3D surface plot
    pyo.plot(fig, filename=f"{fn.__name__}.html")   # <- save the plot to an html file


# %% Evolution as optimization ##########################################
# Implement a simple evolutionary strategy to find the minimum of a
# [function](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
# 1. Select a function and implement it in jax.
# 2. Implement a simple ES algorithm.
# 3. Find the minimum of the function.

def booth_fn(x: Array, y: Array) -> Array:
    return (x + 2 *y + 7)**2 + (2*x + y - 5) **2

# %% 1.
def ackley_fn(x: Array, y: Array) -> Array:
    return -20 * jnp.exp(-0.2 * jnp.sqrt(0.5 * (x**2 + y**2))) - jnp.exp(0.5 * (jnp.cos(2 * jnp.pi * x) + jnp.cos(2 * jnp.pi * y))) + jnp.e + 20

# %% 2.
def init_population(rng, size, dim):  # for 2d optimization problems size=100, dim=2
    return random.normal(rng, (size, dim))

def mutate(rng, parents, std=0.1):
    return parents + random.normal(rng, parents.shape) * std

def evaluate(fn, population):
    return fn(population[:, 0], population[:, 1])

def get_elites(pop, fitness, cutoff):
    sorted_idxs = fitness.argsort()
    ranking = pop[sorted_idxs]
    elites = ranking[:cutoff]
    return elites

# %% Basic Neuroevolution ###############################################
# Take the code from the previous weeks, and replace the gradient
# descent with your ES algorithm.
size = 100
dim = 2
elite_cutoff = 10
iterations = 2000

pop = init_population(rng, size=size, dim=dim)
for i in range(iterations):
    rng, key = random.split(rng)
    fitness = evaluate(ackley_fn, pop)
    # fitness = 1 / fitness (this can be used to invert the fitness, depending on going for minimum or maximum)
    elites = get_elites(pop, fitness, elite_cutoff)
    new_pop = elites.repeat(10, axis=0)
    pop = mutate(rng, new_pop)

sorted_idxs = fitness.argsort()
ranking = pop[sorted_idxs]
best_5 = ranking[:5]
print(best_5)
print(ackley_fn(best_5[0][0], best_5[0][1]))

# %% (Bonus) Growing topologies #########################################
# Implement a simple genetic algorithm to evolve the topology of a
# neural network. Start with a simple network and evolve the number of
# layers, the number of neurons, and the activation functions.
