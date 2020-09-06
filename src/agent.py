# Dependencies
import scipy.special as sp
import numpy as np


class Agent(object):

    def __init__(self, num_states, num_actions, max_reward=1.0, prc_discount=0.9, use_softmax=False, use_sarsa=False):
        """ Constructor

        Args
        num_states              Number of available states (cells)
        num_actions             Number of actions (movements) to perform
        max_reward (float)      Maximum reward value to apply
        prc_discount (float)    Discount percentage to apply
        use_softmax (bool)      Whether to apply Softmax or not
        use_sarsa (bool)        Whether to use SARSA approach or not
        """
        # Store number of available states
        self.num_states = num_states
        # Store number of available actions
        self.num_actions = num_actions
        # Store reward factor
        self.max_reward = max_reward
        # Store discount factor
        self.prc_discount = prc_discount
        # Define whether to apply softmax or not
        self.use_softmax = bool(use_softmax)
        # Define whether to use SARSA approach
        self.use_sarsa = bool(use_sarsa)

        # Initialize qtable
        self._qtable = np.ones((num_states, num_actions), dtype=np.float)
        # Scale it according to given max reward value and discount percentage
        self._qtable *= max_reward / (1 - prc_discount)

    @property
    def qtable(self):
        return self._qtable

    def choose_action(self, state, epsilon=0.0):
        """ Probabilistic random choice of an action

        Args
        state (int)         Index of the actual state
        epsilon (float)     ???
        """
        # For current state, retrieve possible actions
        qval = self.qtable[state, :]
        # Initialize probabilities array
        prob = np.zeros_like(qval, dtype=np.float)

        # Case softmax is set
        if self.softmax:
            # Compute probabilities according to softmax
            prob = sp.softmax(qval / epsilon)

        # Otherwise
        else:
            # Assign equal probabilities to each action
            prob = np.ones(self.num_actions) * epsilon / (self.num_actions - 1)
            # Set best action probability to 1 - epsilon
            prob[np.argmax(qval)] = 1 - epsilon

        # Return an action index at random, using computed probabilities
        return np.random.choice(range(0, self.num_actions), p=prob)

    def update_state(self, state, action, reward, next_state, alpha=0.0, epsilon=0.0):
        """ Update agent's state

        """
        # Define epsilon (only for SARSA setting)
        epsilon = epsilon if self.use_sarsa else 0.0
        # Find next action (use epsilon only for SARSA)
        next_action = self.choose_action(state=next_state, epsilon=epsilon)

        # Compute long-term reward with bootstrap method
        observed = reward + self.prc_discount * self.qtable[next_state, next_action]
        # Make bootstrap update to qtable
        self.qtable[state, action] = self.qtable[state, action] * (1 - alpha) + observed * alpha
