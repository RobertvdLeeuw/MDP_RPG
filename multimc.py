from core import ChooseActionEpsilon, CreateBoltzmannPolicy, GenerateMDP
from multimdp import Agent, GenerateMDP, State

from collections import defaultdict
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.signal import savgol_filter


alpha = 0.25  # Learning rate
gamma = 1  # Discount factor between (0-1)
episodes = 5
T = 100  # Maximum steps in an episode
temp = 0.01

mdp = GenerateMDP(maxLevel=5,
                  costOfLiving=0,
                  agents=2,
                  startingState=State(level=1, state='rested'))


def UpdateQ(value: float, Gt: float) -> float:
    return value + alpha * (Gt - value)


def Update(episode: list, agent: Agent) -> dict:
    G = 0
    for state, action, reward in reversed(episode):
        G = reward + gamma * G
        agent.qTable[state][action] = UpdateQ(agent.qTable[state][action], G)

    agent.totalRewards.append(sum([r for s, a, r in episode]))
    return CreateBoltzmannPolicy(agent.qTable, agent.temp)


for a in mdp.agents:
    a.qTable = {s: {a: 0 for a in mdp.agents[0].getActions(s)} for s in mdp.states}
    a.temp = temp
    a.policy = CreateBoltzmannPolicy(a.qTable, a.temp)

for _ in range(episodes):
    episode = defaultdict(lambda: [])

    terminated = False
    for t in range(T):
        if terminated:
            for a in mdp.agents:
                a.reset()
            break

        actions = {}

        for a in mdp.agents:
            actions[a] = ChooseActionEpsilon(a.policy[a.currentState], 0)

            _, reward, _, truncated, info = a.step(actions[a], message=False)
            episode[a].append((a.prevState, actions[a], reward))

        # Compare
        nextActions = {}

        if 'attack' in actions.values():
            attacker = [a for a, s in actions.items() if s == 'attack']

            if len(attacker) == 2:  # Both attacking
                nextActions[attacker[0]] = 'tired'
                nextActions[attacker[1]] = 'tired'

            else:
                attacker = attacker[0]
                other = [a for a in mdp.agents if a != attacker][0]

                match actions[other]:
                    case 'rest':
                        nextActions[other] = 'die'
                        nextActions[attacker] = 'win'

                    case 'train':
                        nextActions[other] = 'die'
                        nextActions[attacker] = 'win'

                    case 'defend':
                        nextActions[other] = 'tired'
                        nextActions[attacker] = 'tired'
        else:
            for a, next_state in actions.items():
                match next_state:
                    case 'rest':
                        nextActions[a] = 'rest'

                    case 'train':
                        nextActions[a] = 'train'

                    case 'defend':
                        nextActions[a] = 'nothing'


        for a in mdp.agents:
            _, reward, terminated, truncated, info = a.step(nextActions[a], message=False)
            episode[a].append((a.prevState, nextActions[a], reward))

    for a in mdp.agents:
        a.policy = Update(episode[a], a)

    pprint(episode)


def Plot() -> None:
    for a in mdp.agents:
        # moving_averages = savgol_filter(a.totalRewards, 35, 3)
        # plt.plot(moving_averages, label=f'Total Reward)')
        plt.plot(a.totalRewards, label=f'Total Reward)')
    plt.title(f'Rewards per Episode (alpha={alpha}, gamma={gamma})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


Plot()
