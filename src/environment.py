# Dependencies
import matplotlib.colors as col
import matplotlib.pyplot as plt
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
        'right': [1, 0],
        # Go left one cell
        'left': [-1, 0]
    }

    # Define plot color map (from higher to lower reward)
    cmap = col.ListedColormap(['white', 'tab:cyan', 'tab:orange', 'tab:red', 'tab:gray'])

    def __init__(self, width, height, start, goal, walls=[], holes=[], sands=[]):
        """ Constructor

        Args
        width (int)         Width of the environemnt to create
        height (int)        Height of the environment to create
        start (iterable)    Initial position (x, y) of the agent
        goal (iterable)     Goal position (x, y) of the agent
        holes (list)        List of holes, i.e. cells which cannot be accessed
        sands (list)        List of sand cells, i.e. cells with lower reward
        """
        # Define shape of the environment (boundaries)
        self.shape = np.array((width, height), dtype=np.int)
        # Define current state (initial agent state)
        self.state = np.array(start, dtype=np.int)
        # Define gaol state (where agent must go)
        self.goal = np.array(goal, dtype=np.int)

        # Store walls as 2d matrix (index, (x, y))
        self.walls = np.array(walls, dtype=np.int).reshape(-1, 2)
        # Store holes as 2d matrix (index, (x, y))
        self.holes = np.array(holes, dtype=np.int).reshape(-1, 2)
        # Store sands as 2d matrix (index, (x, y))
        self.sands = np.array(sands, dtype=np.int).reshape(-1, 2)

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

        # Case agent is in a hole
        if np.all(self.holes == self.state, axis=1).any():
            # Set minimum reward possible
            reward = -1.0
        # Case next state is outside boundaries
        elif (next_state < 0).any() or (self.shape <= next_state).any():
            # Set reward to minimum possible (go back at worse)
            reward = -1.0
        # Case next state is a hole (cannot be traversed)
        elif np.all(self.walls == next_state, axis=1).any():
            # Set reward to minimum possible (go back at worse)
            reward = -1.0
        # Otherwise
        else:
            # Case next state is a hole
            if np.all(self.holes == next_state, axis=1).any():
                # Set minimum reward possible
                reward = -1.0
            # Case new state is sand (can be traversed with penalty)
            elif np.all(self.sands == next_state, axis=1).any():
                # Set half of the reward
                reward = -0.5
            # Update corrent state with next state
            self.state = next_state

        # Return both current state and reward
        return (self.state, reward)

    def plot(self, cell_size=5, font_size=18):
        """ Plot the environment

        Args
        cell_size (float)   Size of a single cell plotted
        font_size (float)   Size of the font used in labels

        Return
        (plt.Axis)          Plotted environment
        """
        # Retrieve width and height
        w, h = tuple(self.shape)
        # Define the environment matrix to show
        env = np.zeros(shape=(w, h), dtype=np.int)

        # Set sands (orange)
        for i in range(self.sands.shape[0]):
            # Define sand coordinates
            x, y = tuple(self.sands[i, :])
            # Set sand at current coordinate
            env[x, y] = 2

        # Set holes (red)
        for i in range(self.holes.shape[0]):
            # Define hole coordinates
            x, y = tuple(self.holes[i, :])
            # Set hole at current coordinate
            env[x, y] = 3

        # Set walls (black)
        for i in range(self.walls.shape[0]):
            # Define wall coordinates
            x, y = tuple(self.walls[i, :])
            # Set hole at current coordinate
            env[x, y] = 4

        # Retrieve goal point
        goal = self.goal
        # Set goal at given coordinates
        env[goal[0], goal[1]] = 1

        # Initialize figure
        fig = plt.figure(figsize=(w * cell_size, h * cell_size))
        # Plot environment matrix
        img = plt.imshow(env.T, vmin=0, vmax=len(self.cmap.colors), interpolation='none', cmap=self.cmap)
        # Initialize axis
        ax = plt.gca()
        # Define x and y ticks
        x_ticks, y_ticks = np.arange(w), np.arange(h)
        # Reset x and y axis major ticks positions
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        # Set x and y axis major ticks labels
        ax.set_xticklabels(x_ticks, fontsize=font_size)
        ax.set_yticklabels(y_ticks, fontsize=font_size)

        # Define x and y axis minor ticks positions
        ax.set_xticks(x_ticks - 0.5, minor=True)
        ax.set_yticks(y_ticks - 0.5, minor=True)
        # Set girdlines, according to minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

        # # Plot start point
        # ax.text(x=goal[0], y=goal[1], s='GOAL', ha='center', va='center')

        # Put x ticks on top of image
        ax.xaxis.tick_top()

        # # Debug
        # print('Goal:', goal)
        # print('Shape:', (w, h))

        # Return plotted figure and axis
        return fig, ax
