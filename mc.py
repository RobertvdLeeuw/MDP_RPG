from core import ChooseActionEpsilon, GenerateMDP, GenerateRandomPolicy
from gpi import UpdatePolicy
from mdp import State

import matplotlib.pyplot as plt
from pprint import pprint
from scipy.signal import savgol_filter


alpha = 0.25  # Learning rate
gamma = 1  # Discount factor between (0-1)
episodes = 4000
T = 100  # Maximum steps in an episode
epsilon = 0.025  # Exploration rate between (0-1)

mdp = GenerateMDP(maxLevel=5,
                  attackedChance=0,
                  costOfLiving=0,
                  startingState=State(level=1, tired=False))

vTable = {s: 0 for s in mdp.states}
qTable = {s: {a: 0 for a in mdp.getActions(s)} for s in mdp.states}

policy = GenerateRandomPolicy(mdp)


def Update(value: float, Gt: float) -> float:
    return value + alpha * (Gt - value)


totalRewards = []

for _ in range(episodes):
    episode = []
    state = mdp.reset()

    for t in range(T):
        action = ChooseActionEpsilon(policy[state], epsilon)

        next_state, reward, terminated, truncated, info = mdp.step(action, message=False)
        episode.append((state, action, reward))

        if terminated:
            break

        state = next_state

    G = 0
    for state, action, reward in reversed(episode):
        G = reward + gamma * G
        qTable[state][action] = Update(qTable[state][action], G)
        vTable[state] = Update(vTable[state], G)  # Every-visit update

    policy = UpdatePolicy(policy, qTable)
    totalRewards.append(sum([r for s, a, r in episode]))


def Plot(x: list, title: str) -> None:
    moving_averages = savgol_filter(x, 35, 3)
    plt.plot(moving_averages, label=f'Total Reward)', color='orange')
    plt.title(f'Rewards per Episode ({title})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


Plot(totalRewards, f'alpha={alpha}, gamma={gamma}, epsilon={epsilon}')
