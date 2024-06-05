from mdp import MDP, State

from copy import deepcopy
from pprint import pprint


GAMMA = 1
LEARNING_RATE = 0.01


def Gt(state: State, action: str, nextState: State, mdp: MDP, vTable: dict) -> float:
    return mdp.getReward(state, action, nextState) + (GAMMA * vTable[nextState])


def CalculateV(policy: dict, state: State, mdp: MDP, vTable: dict) -> float:
    if mdp.isTerminal(state):
        return 0

    return sum(p * CalculateQ(state, a, mdp, vTable)
               for a, p in policy[state].items())


# expected reward: ∑s′,rp(s′,r∣s,a)[r+γV(s′)]
def CalculateQ(state: State, action: str, mdp: MDP, vTable: dict) -> float:

    return sum(p * Gt(state, action, sNext, mdp, vTable)
               for sNext, p in mdp.transitions[state][action].items())


def UpdateQ(state: State, policy: dict, mdp: MDP, vTable: dict) -> dict:
    return {a: CalculateQ(state, a, mdp, vTable) for a in policy[state]}


def ArgMax(d: dict):  # np.argmax was causing issues.
    for k in d:
        if d[k] == max(d.values()):
            return k


def UpdatePolicy(policy: dict, actionValues: dict) -> dict:  # Choosing the optimal action from our actionValues.
    updatedPolicy = deepcopy(policy)

    for state in policy:
        optimalAction = ArgMax(actionValues[state])

        updatedPolicy[state] = {a: int(a == optimalAction) for a in policy[state]}

    return updatedPolicy


def UpdateV(vTable: dict, policy: dict, mdp: MDP, sweeps: int) -> dict:
    for _ in range(sweeps):
        tableCopy = deepcopy(vTable)

        vTable = {s: CalculateV(policy, s, mdp, tableCopy) for s in tableCopy}

    return vTable


def GPI(iterations: int, policy: dict, mdp: MDP) -> dict:
    vTable = {s: 0 for s in mdp.statesPlus}

    for i in range(iterations):
        # Evaluation
        vTable = UpdateV(vTable, policy, mdp, sweeps=100)
        actionValues = {s: UpdateQ(s, policy, mdp, vTable) for s in mdp.states}

        # Improvement
        policy = UpdatePolicy(policy, actionValues)

    return policy

