from core import ChooseAction, GenerateMDP, GenerateRandomPolicy
from gpi import UpdatePolicy
from mdp import State


alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor between (0-1)
episodes = 4 * 2000  # Number of episodes to learn, keep this a multiple of four for nice plotting
T = 100  # Maximum steps in an episode
epsilon = 0.05  # Exploration rate between (0-1)

mdp = GenerateMDP(maxLevel=5,
                  attackedChance=0,
                  costOfLiving=0,
                  startingState=State(level=1, tired=False))

vTable = {s: 0 for s in mdp.states}
qTable = {s: {a: 0 for a in mdp.getActions(s)} for s in mdp.states}


def Update(value: float, Gt: float) -> float:
    return value + alpha * (Gt - value)


policy = GenerateRandomPolicy(mdp)

for _ in range(episodes):
    episode = []
    state = mdp.reset()

    for t in range(T):
        action = ChooseAction(policy[state], epsilon)

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
