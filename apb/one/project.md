# Mini Project one
### Andreas Pries-Brokmann (prie@itu.dk)
### 16/9/2024

## Structure
The code is split into make_agent.py and run_agent.py.
run_agent.py has a method for both creating and running an agent. 
Make_agent will output a file named agent.pkl which is my trained agent


## My code
I have tried to use deep reinforcement learning to teach my agent how to balance a pole
I have set an experience rate that reduces with the amount of iterations, to ensure that my agent explores early in the episode
I then run my policy with the give nexploration rate and store the state, my action, the reward and the next state in a deque to be learned from later
Once there is suffecient experiences in the deque i use thsoe experiences to teach my mlp, this is done in the train_mlp function

My mlp does not seem to learn/adapt correctly and is thus unable to balance the code any better (if not worse) than a random agent.
I suspect i either have messed up my mlp in some way, be that structure (init_mlp) or using the mlp (run_mlp) or that my loss function is wrong, though i have been unable to identify what i have done wrong.
