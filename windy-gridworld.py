from collections import defaultdict
from functools import total_ordering
import matplotlib.pyplot as plt
import random
import logging


@total_ordering
class State:
    MAX_X = 9
    MAX_Y = 6

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y
        else:
            return False

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"({self.x},{self.y})"

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)


class Action:
    @staticmethod
    def up(state):
        return State(state.x, min(State.MAX_Y, state.y + 1))

    @staticmethod
    def down(state):
        return State(state.x, max(0, state.y - 1))

    @staticmethod
    def left(state):
        return State(max(0, state.x - 1), state.y)

    @staticmethod
    def right(state):
        return State(min(State.MAX_X, state.x + 1), state.y)


class DiagonalAction(Action):
    @staticmethod
    def up_left(state):
        return State(max(0, state.x - 1), min(State.MAX_Y, state.y + 1))

    @staticmethod
    def up_right(state):
        return State(min(State.MAX_X, state.x + 1), min(State.MAX_Y, state.y + 1))

    @staticmethod
    def down_left(state):
        return State(max(0, state.x - 1), max(0, state.y - 1))

    @staticmethod
    def down_right(state):
        return State(min(State.MAX_X, state.x + 1), max(0, state.y - 1))


class DiagonalOrStayAction(DiagonalAction):
    @staticmethod
    def stay(state):
        return state


class Rewards:
    def __init__(self, goal):
        self.goal = goal

    def get(self, state):
        if state == self.goal:
            return 1
        else:
            return -1


class Environment:
    def __init__(self, rewards):
        self.rewards = rewards
        self.winds = defaultdict(lambda: (None, 0))

        logging.info(f"Environment created with goal={rewards.goal}")

    def register_wind(self, state, action, strength):
        self.winds[state] = (action, strength)

    def apply_wind(self, state):
        action, strength = self.winds[state]
        for _ in range(strength):
            state = action(state)
        return state

    def take_action(self, state, action):
        logging.debug(f"Taking action '{action.__name__}' from {state}")
        # Apply initial movement
        new_state = action(state)
        logging.debug(f"Moved to {new_state}")
        # Correct position applying winds
        new_state = self.apply_wind(new_state)
        logging.debug(f"Corrected to {new_state} by wind")
        # Get the reward
        reward = self.rewards.get(new_state)

        return new_state, reward


class Q:
    def __init__(self, default):
        self.default = float(default)
        self.q = {}

    def get(self, state, action):
        return self.q.get((state, action), self.default)

    def set(self, state, action, value):
        self.q[(state, action)] = value

    def best_action(self, state):
        values = [(k[1], v) for k, v in self.q.items() if k[0] == state]
        return max(values, key=lambda x: x[1])[0] if values else None

    def __str__(self):
        items = [(k[0], k[1].__name__, v) for k, v in self.q.items()]
        items.sort()
        return "\n".join([f"Q({s}, {a})\t= {v}" for s, a, v in items])


class Agent:
    def __init__(self, epsilon, alpha, gamma, start, action_class):
        for x in [epsilon, alpha, gamma]:
            assert x < 1.0 and x > 0.0

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.start = start

        self.action_class = action_class
        self.q = Q(0)
        self.stats = []

        self._restart_episode()

        logging.info(
            f"Creating agent with state {start} and action {self.action.__name__}"
        )

    def _restart_episode(self):
        self.state = self.start
        self.action = self._select_action(self.state)
        self.trajectory = [(self.state, self.action.__name__)]

    def _random_action(self):
        def filter(x):
            return (
                callable(getattr(self.action_class, x)) and x.startswith("__") is False
            )

        actions = [
            getattr(self.action_class, x) for x in dir(self.action_class) if filter(x)
        ]
        return random.choice(actions)

    def _select_action(self, state):
        r = self._random_action()

        if random.random() < self.epsilon:
            logging.debug(f"Selecting random action")
            return r
        else:
            logging.debug(f"Greedily selecting action")
            best_action = self.q.best_action(state)
            return best_action if best_action is not None else r

    def _update(self, environment):
        S = self.state
        A = self.action

        S_prime, R = environment.take_action(S, A)
        A_prime = self._select_action(S_prime)
        logging.debug(
            f"Taking action '{A.__name__}' from {S} to {S_prime} with reward {R}"
        )

        Q = self.q.get(S, A)
        Q_prime = self.q.get(S_prime, A_prime)

        update = Q + self.alpha * (R + self.gamma * Q_prime - Q)
        self.q.set(S, A, update)
        logging.debug(f"Updating Q({S}, {A.__name__}) = {update}")

        self.state = S_prime
        self.action = A_prime
        self.trajectory += [(self.state, self.action.__name__)]
        logging.debug(f"New state is {self.state} and action is {self.action.__name__}")

    def run(self, environment, goal, episodes, steps, epsilon=None):
        if epsilon is not None:
            self.epsilon = epsilon

        for episode in range(episodes):
            logging.debug(f"Running episode {episode}")
            self._restart_episode()

            for step in range(steps):
                self._update(environment)
                if self.state == goal:
                    if episode % 100 == 0:
                        logging.info(f"Episode {episode}: Goal reached in {step} steps")
                    break

            self.stats += [(episode, step)]
            if self.state != goal:
                logging.warning(f"Episode {episode}: Goal not reached in {steps} steps")

    def plot(self, environment, figure=None, start=None, goal=None):
        if figure is None:
            figure = plt.figure(figsize=(10, 10))

        # Plot training stats
        plt.subplot(2, 1, 1)
        plt.plot(*zip(*self.stats), label=self.action_class.__name__)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.legend()

        # Plot trajectory
        plt.subplot(2, 1, 2)
        values = [(state.x, state.y) for state, _ in self.trajectory]
        plt.plot(
            *zip(*values),
            ".-",
            label=self.action_class.__name__,
        )
        if start is not None:
            plt.plot(self.start.x, self.start.y, "o", label="Start")
        if goal is not None:
            plt.plot(goal.x, goal.y, "o", label="Goal")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()

        for state, (_, strength) in environment.winds.items():
            if strength > 0:
                plt.scatter(state.x, state.y, s=strength * 100, c="r", alpha=0.5)

        # Save figure
        plt.savefig("windy-gridworld.png")

    def __str__(self):
        trajectory = [f"({S}, {A})" for S, A in self.trajectory]
        return f"\n\nFinal State: {self.state}\n\n{trajectory}\n"

    def __repr__(self):
        return f"\n\nFinal State: {self.state}\n\n{self.q}\n"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    start = State(0, 3)
    goal = State(7, 3)

    rewards = Rewards(goal)
    environment = Environment(rewards)
    [environment.register_wind(State(3, y), Action.up, 1) for y in range(7)]
    [environment.register_wind(State(4, y), Action.up, 1) for y in range(7)]
    [environment.register_wind(State(5, y), Action.up, 1) for y in range(7)]
    [environment.register_wind(State(6, y), Action.up, 2) for y in range(7)]
    [environment.register_wind(State(7, y), Action.up, 2) for y in range(7)]
    [environment.register_wind(State(8, y), Action.up, 1) for y in range(7)]

    episodes, steps = 5_000, 5_000
    epsilon, alpha, gamma = 0.1, 0.5, 0.1

    figure = plt.figure(figsize=(10, 10))

    agent = Agent(epsilon, alpha, gamma, start, Action)
    agent.run(environment, goal, episodes, steps)
    agent.run(environment, goal, 1, steps, epsilon=0.01)
    agent.plot(environment, figure, start, goal)
    print(agent)

    # agent = Agent(epsilon, alpha, gamma, start, DiagonalAction)
    # agent.run(environment, goal, episodes, steps)
    # agent.run(environment, goal, 1, steps, epsilon=0.00)
    # agent.plot(environment, figure)
    # print(agent)

    # agent = Agent(epsilon, alpha, gamma, start, DiagonalOrStayAction)
    # agent.run(environment, goal, episodes, steps)
    # agent.run(environment, goal, 1, steps, epsilon=0.00)
    # agent.plot(environment, figure)
    # print(agent)
