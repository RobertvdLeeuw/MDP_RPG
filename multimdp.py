from math import isclose

from frozendict import frozendict
from random import choice, choices
from pprint import pprint


class State(frozendict):
    def __new__(cls, *args, **kwargs):
        return frozendict.__new__(cls, *args, **kwargs)

    def __str__(self):
        if 'title' in self:
            return self['title']

        return f"L{self['level']} ({self['state']})"

    def __repr__(self):
        return self.__str__()


class Agent:
    def __init__(self, startingState=None):
        self.startingState = startingState
        self.currentState: State = self.startingState
        self.prevState: State
        self.stepsTaken = 0

        self.mdp: MultiMDP

        self.qTable: dict
        self.policy: dict

        self.temp: float

        self.totalRewards = []

    def reset(self) -> None:
        self.prevState, self.currentState = None, self.startingState
        self.stepsTaken = 0


    def getActions(self, state=None) -> list:
        if state is None:
            state = self.currentState

        return self.mdp.actions[state]

    def step(self, action, message=False) -> tuple:
        if action not in self.mdp.actions[self.currentState]:
            print('\tInvalid action:', action)
            return (self.currentState,
                    0,
                    False,
                    False,
                    "Invalid")

        possibleOutcomes = self.mdp.transitions[self.currentState][action]  # {s: chance, s: chance, etc.}
        nextState = choices(list(possibleOutcomes.keys()), list(possibleOutcomes.values()))[0]

        reward = self.mdp.getReward(self.currentState, action, nextState)
        # print(self.currentState, action, nextState, reward)

        if message:
            print(f"\t{self.currentState} -> {nextState} via {action} (reward of {reward:.2f})")

        self.stepsTaken += 1
        self.prevState, self.currentState = self.currentState, nextState

        return (nextState,
                reward,
                self.mdp.isTerminal(nextState),
                # In our case we treat 'sinks' (states with no further transitions) as terminal.
                self.stepsTaken >= self.mdp.T,
                None)  # What should info be?

    def __str__(self):
        return 'Agent'

    def __repr__(self):
        return str(self)




class MultiMDP:
    def __init__(self, states, statesPlus, actions, agents: list[
        Agent], transitions=None, rewards=None, gamma=0.9, eps=1e-6, T=100, costOfLiving: float =0):
        self.states = states
        self.statesPlus = statesPlus
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
        self.eps = eps
        self.T = T
        self.costOfLiving = costOfLiving

        self.agents = agents
        for a in agents:
            a.mdp = self

        self.checkProbabilities()

    def checkProbabilities(self):
        for s, a in self.transitions.items():
            for action, outcomes in a.items():
                try:
                    assert isclose(sum(outcomes.values()), 1,
                                   abs_tol=1e-4)  # Making sure the sum of outcomes is effectively 1
                except AssertionError:
                    raise AssertionError(f"Probability not 1: {s} + {action} -> {outcomes.values()}")

                for o, chance in outcomes.items():
                    try:
                        assert 0 <= chance <= 1
                    except AssertionError:
                        raise AssertionError(f"Invalid probability 1: {s} + {action} -> {o} ({chance})")

    def reset(self):
        for a in self.agents:
            a.currentState = a.startingState
            a.stepsTaken = 0

    @property
    def stateSpace(self) -> int:
        return len(self.states)

    @property
    def allActions(self) -> set:
        return set([a for A in self.actions.values() for a in A])

    @property
    def actionSpace(self) -> int:
        return len(set(self.allActions))  # To set to remove duplicate nextActions

    def isTerminal(self, state):
        return state not in self.transitions

    def getReward(self, state, action, newState):
        return self.rewards.get(state, {}).get(action, {}).get(newState, 0) + self.costOfLiving


def ActionsFromTransitions(t: dict) -> dict:  # state: {action1: outcomes, action2: outcomes} -> state: [nextActions]
    return {s: list(A) for s, A in t.items()}


def StatesFromTransitions(t: dict) -> list:
    return list(t)


def To(s: State, state: str, level: int = 0) -> State:
    return State(level=s['level'] + level,
                 state=state)



def WinChance(s: State, maxLevel: int, invert: bool = False) -> float:
    winChance = 0.1 + (0.9 / (maxLevel - 1)) * (s['level'] - 1)

    if s['tired']:
        winChance /= 2

    return (1 - winChance) if invert else winChance


def GenerateMDP(maxLevel: int, agents: int, costOfLiving: float = 0, startingState: State = None) -> MultiMDP:
    states = [State(level=l, state=s) for l in range(1, maxLevel + 1) for s in ('rested', 'tired', 'attacking', 'defending', 'resting')]
    states += [State(level=l, state='training') for l in range(1, maxLevel)]

    won, died = State(title="Won"), State(title="Died")
    statesPlus = states + [won, died]

    transitions = {}

    for s in states:
        match s['state']:
            case 'tired':
                t = {'attack': {To(s, 'attacking'): 1}, 'rest': {To(s, 'resting'): 1}}

            case 'rested':
                t = {'attack': {To(s, 'attacking'): 1},
                     'train': {To(s, 'training'): 1},
                     'defend': {To(s, 'defending'): 1}}

            case 'resting':
                t = {'die': {died: 1},
                     'rest': {To(s, 'rested'): 1}}

            case 'training':
                t = {'die': {died: 1},
                     'train': {To(s, 'tired', 1): 1}}

            case 'defending':
                t = {'tired': {To(s, 'tired'): 1},
                     'nothing': {To(s, 'rested'): 1}}

            case 'attacking':
                t = {'die': {died: 1},
                     'win': {won: 1},
                     'tired': {To(s, 'tired'): 1}}

        transitions[s] = t


    rewards = {s: {'die': {died: -1},
                   'win': {won: 1}}
               for s in states}


    return MultiMDP(states,
                    statesPlus,
                    ActionsFromTransitions(transitions),
                    agents=[Agent(startingState) for _ in range(agents)],
                    transitions=transitions,
                    rewards=rewards,
                    costOfLiving=costOfLiving)
