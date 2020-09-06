# Dependencies
import numpy as np


class Environment(object):

    # Define available actions
    # NOTE Actions are vectors added to current state for movement
    actions = {
        # Do not move from current cell
        'stay': [0, 0],
        # Go up one cell
        'up': [0, 1],
        # Go down one cell
        'down': [0, -1],
        # Go right one cell
        'right': [0, 1],
        # Go left one cell
        'left': [0, -1]
    }

    def __init__(self, width, height, start, goal):
        """ Constructor

        Args
        width (int)         Width of the environemnt to create
        height (int)        Height of the environment to create
        start (iterable)    Initial position (x, y) of the agent
        goal (iterable)     Goal position (x, y) of the agent
        """
        # Define shape of the environment (boundaries)
        self.shape = (width, height)
        # Define current state (initial agent state)
        self.state = np.array(start)
        # Define gaol state (where agent must go)
        self.goal = np.array(goal)

    def move(self, action):
        """ Move from current state

        Args
        action (str)            Movement to apply

        Return
        (np.array)              New cell of the agent
        (float)                 Reward obtained from movement
        """
        # Initialize reward
        reward = 0.0
        # Define movement
        movement = np.array(self.actions[action])
        # Case goal has been reached
        if (action == 'stay') and (self.state == self.goal).all():
            # Set reward to 1
            reward = 1.0

        # Define next state
        next_state = self.state + movement
        # Case next state is outside boundaries
        if not (0 < next_state < self.shape).any():
            # Set reward to minimum possible (go back at worse)
            reward = -1.0
        # Otherwise
        else:
            # Update corrent state with next state
            self.state = next_state

        # Return both current state and reward
        return (self.state, reward)
