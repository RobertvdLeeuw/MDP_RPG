from core import ChooseActionEpsilon, CreateBoltzmannPolicy, GenerateMDP, GenerateRandomPolicy
from mdp import State

import numpy as np
from pprint import pprint


alpha = 0.25  # Learning rate
gamma = 1  # Discount factor between (0-1)
episodes = 100000
T = 100  # Maximum steps in an episode
temp = 0.025

mdp = GenerateMDP(maxLevel=5,
                  attackedChance=0.4,
                  costOfLiving=0,
                  startingState=State(level=1, tired=False))

vTable = {s: 0 for s in mdp.states}
qTable = {s: {a: 0 for a in mdp.getActions(s)} for s in mdp.statesPlus}

policy = GenerateRandomPolicy(mdp)


def Update(value: float, nextState, terminated: bool) -> float:
    if terminated:
        return value + alpha * (reward - value)

    return value + alpha * (reward + gamma * np.max(list(qTable[nextState].values())) - value)


totalRewards = []

for _ in range(episodes):
    episode = []
    state = mdp.reset()

    for t in range(T):
        action = ChooseActionEpsilon(policy[state], 0)
        # TODO: SOL, action = CreateBoltzmannPolicy(qTable, temp)

        next_state, reward, terminated, truncated, info = mdp.step(action, message=False)

        qTable[state][action] = Update(qTable[state][action], next_state, terminated)

        if terminated:
            break

        episode.append((state, action, reward))
        state = next_state
        policy = CreateBoltzmannPolicy(qTable, temp)

pprint(qTable)
