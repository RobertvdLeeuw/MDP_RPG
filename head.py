from core import GenerateMDP, GenerateRandomPolicy, Plot, Simulate
from gpi import GPI

from pprint import pprint


# MDP Settings
ATTACKED_CHANCE = 0.1
MAX_LEVEL = 5
COST_OF_LIVING = -0

mdp = GenerateMDP(MAX_LEVEL, ATTACKED_CHANCE, COST_OF_LIVING)

# Episode settings
EPISODES = 4000
STEPS = 100



# Plot(Simulate(GenerateRandomPolicy(mdp), mdp, EPISODES, STEPS)[0], 'Random')
Simulate(GenerateRandomPolicy(mdp), mdp, EPISODES, STEPS)

policy = GPI(10, GenerateRandomPolicy(mdp), mdp)
# Plot(Simulate(GenerateRandomPolicy(mdp), mdp, EPISODES, STEPS)[0], 'After GPI')
