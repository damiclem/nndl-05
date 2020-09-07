# Dependencies
from src.environment import Environment
from src.agent import Agent
from tempfile import mkdtemp
import matplotlib.pyplot as plt
import numpy as np
import argparse
import datetime
import imageio
import shutil
import glob
import dill
import os


def to_squared(i, width, height=None):
    """ Compute squared indices from linearized one

    Args
    i (int)         Linearized input index
    width (int)     Width of the matrix
    height (int)    Height of the matrix

    Return
    (list)          Coordinates for squared matrix
    """
    # Set height equal to width, if not set
    height = width if height is None else height
    # Get row index
    y = i // width
    # Get column index
    x = i - y * width
    # Return coordinates
    return [x, y]


def make_gif(in_path, out_path):
    """ Make a GIF out of PNG imagesin given folder

    Args
    in_path (str)       Path to directory where images can be found
    out_path (str)      Path to output GIF animated image
    """
    # Retrieve all png files from current directory
    in_paths = sorted(glob.glob(os.path.join(in_path, '*.png')))
    # Define list of images
    images = [imageio.imread(in_path) for in_path in in_paths]
    # Save gif
    imageio.mimsave(out_path, images, fps=2)


# Main
if __name__ == '__main__':

    # Define now datetime
    now = datetime.datetime.now()

    # Define arument parser
    parser = argparse.ArgumentParser('Train a new agent')
    # Add argument: number of training episodes
    parser.add_argument(
        '--num_episodes', type=int, default=1000,
        help='Define number of training episodes'
    )
    # Add argument: length of each training episode
    parser.add_argument(
        '--len_episodes', type=int, default=10,
        help='Define length of each training episode'
    )
    # Add argument: heigh of the environment
    parser.add_argument(
        '--env_height', type=int, default=10,
        help='Define height of the environment'
    )
    # Add argument: width of the environemnt
    parser.add_argument(
        '--env_width', type=int, default=10,
        help='Define width of the environment'
    )
    # Add argument: starting point
    parser.add_argument(
        '--start_point', type=int, required=False,
        help='Define a starting point for the agent'
    )
    # Add argument: goal point
    parser.add_argument(
        '--goal_point', type=int, required=False,
        help='Define a goal point to reach for the agent'
    )
    # Add argument: wall points
    parser.add_argument(
        '--wall_points', type=int, nargs='*',
        default=[2, 17, 32, 47, 62, 65, 80, 95, 110, 125, 112, 113],
        help='Define points which cannot be reached from the agent'
    )
    # Add argument: hole points
    parser.add_argument(
        '--hole_points', type=int, nargs='*',
        default=[
            135, 136, 137, 138, 139, 140, 141, 142,
            10, 24, 38, 55, 69, 83
        ],
        help='Define points from which agent cannot come out'
    )
    # Add argument: sand points
    parser.add_argument(
        '--sand_points', type=int, nargs='*',
        default=[
            30, 45, 60, 75, 90, 100, 49, 50, 51,
            100, 101, 102, 103, 115, 116, 117, 118, 130, 131, 132, 133
        ],
        help='Define points where agent gets penalty'
    )
    # Add argument: discount percentage
    parser.add_argument(
        '--prc_discount', type=float, default=0.9,
        help='Discount percentage to apply'
    )
    # Add argument: whether to use softmax or not
    parser.add_argument(
        '--use_softmax', type=int, default=0,
        help='Whether to use softmax when choosing for next action'
    )
    # Add argument: whether to use SARSA approach or not
    parser.add_argument(
        '--use_sarsa', type=int, default=0,
        help='Whether to use SARSA approach or not'
    )
    # Add argument: save after some episodes
    parser.add_argument(
        '--save_after', type=int, default=10,
        help='Define when to save trained agent'
    )
    # Add argument: save to defined folder
    parser.add_argument(
        '--agent_path', type=str, default='data/%s' % now.strftime('%Y_%m_%d_%H_%M_%S'),
        help='Path where to save trained model'
    )
    # Add argument: whether to make trace gif
    parser.add_argument(
        '--make_gif', type=int, default=0,
        help='Whether to make  trace GIF or not'
    )
    # Add argument: verbose log
    parser.add_argument(
        '--verbose', '-v', type=int, default=1,
        help='Whether to print out verbose log or not'
    )
    # Parse arguments
    args = parser.parse_args()

    # Case output directory is set
    if args.agent_path is not None:
        # Make directory
        os.makedirs(args.agent_path, exist_ok=True)

    # Define available actions
    actions = [*Environment.actions.keys()]

    # Define alpha factors
    alpha = np.ones(args.num_episodes, dtype=np.float) * 0.50
    # Define epsilon factor
    epsilon = np.linspace(0.8, 0.001, args.num_episodes, dtype=np.float)

    # Initialize the agent
    agent = Agent(
        num_states=(args.env_width * args.env_height),
        num_actions=len(actions)
    )

    # Define all available points
    available = range(args.env_width * args.env_height)
    # Define starting indices which are not walls
    accessible = set(available) - set(args.wall_points)

    # Define goal point at random
    _goal_point = np.random.choice(sorted(accessible))
    # Case goal point has been defined by the user
    if args.goal_point is not None:
        # Use user defined goal point
        _goal_point = args.goal_point

    # Retrieve wall points list
    walls = [to_squared(i, args.env_width, args.env_height) for i in args.wall_points]

    # Retrieve hole points list
    holes = [to_squared(i, args.env_width, args.env_height) for i in args.hole_points]

    # Retrieve sand points list
    sands = [to_squared(i, args.env_width, args.env_height) for i in args.sand_points]

    # Initialize list of (mean) rewards
    rewards = []
    # Loop through each training episode
    for episode in range(args.num_episodes):

        # Initialize reward
        reward = 0.0

        # Sample start point index at random (excluding goal point)
        _start_point = np.random.choice(sorted(accessible - {_goal_point}))
        # Case starting point has not been defined by the user
        if args.start_point is not None:
            # Use user defined starting point
            _start_point = args.start_point

        # Define both starting point and goal point coordinates
        goal_point = to_squared(_goal_point, args.env_width, args.env_height)
        start_point = to_squared(_start_point, args.env_width, args.env_height)

        # Define current state as starting point
        state = start_point
        # Define new environment
        env = Environment(
            width=args.env_width,
            height=args.env_height,
            start=start_point,
            goal=goal_point,
            walls=walls,
            holes=holes,
            sands=sands
        )

        # Initialize list of cells visited, store first one
        trace = [state]
        # Loop through each step in current training episode
        for step in range(args.len_episodes):

            # Define current cell (flattened) index
            _state = state[0] * args.env_height + state[1]
            # Choose an action (index)
            _action = agent.choose_action(_state, epsilon[episode])
            # Get choosen action
            action = actions[_action]
            # Retrieve next state and reward
            next_state, _reward = env.move(action)
            # Flatten next state index
            _next_state = next_state[0] * args.env_height + next_state[1]
            # Q-learning update
            agent.update_state(_state, _action, _reward, _next_state, alpha[episode], epsilon[episode])
            # Update state
            state = next_state
            # Store new state
            trace += [next_state]
            # Update reward
            reward += _reward

        # Update reward (compute average)
        reward /= args.len_episodes
        # Store reward
        rewards += [reward]

        # Save the agent
        if ((episode + 1) % args.save_after == 0):
            # Open output file
            with open(args.agent_path + '/agent.obj', 'wb') as agent_file:
                # Write out to file
                dill.dump(agent, agent_file)

        # Verbose log
        if args.verbose:
            # Define output message
            msg = 'Episode %04d: ' % (episode + 1)
            msg += 'the agent has obtained an average reward '
            msg += 'of %.03f ' % reward
            msg += 'starting from position (%d, %d)' % tuple(start_point)
            # Print output message
            print(msg)

        # Turn list of states to numpy array
        trace = np.array(trace, dtype=np.float).reshape(-1, 2)
        # Move points at random around the center
        trace += np.random.uniform(-0.25, 0.25, trace.shape[0] * trace.shape[1]).reshape(-1, 2)

    # Initialize temporary plot directory
    os.makedirs(args.agent_path + '/trace', exist_ok=True)
    # Loop through each trace
    for i in range(1, trace.shape[0]):
        # Case no GIF is required an current on is not last iteration
        if (not args.make_gif) and (i + 1) != trace.shape[0]:
            # Skip and go to next iteration
            continue

        # Plot initial environment
        fig, ax = env.plot(cell_size=2)
        # Add route: scatterplot
        ax.plot(trace[1:i, 0], trace[1:i, 1], ls='', marker='o', markersize=24, color='tab:blue')
        # Add initial point: scatterplot
        ax.plot(trace[[0], 0], trace[[0], 1], ls='', marker='X', markersize=24, color='tab:blue')
        # Store image
        plt.savefig(args.agent_path + '/trace/%09d.png' % i, dpi=20)
        # Close plot
        plt.close()

    # Case GIF is required
    if args.make_gif:
        # Make GIF
        make_gif(args.agent_path + '/trace', args.agent_path + '/trace.gif')

    # Plot mean reward
    fig, ax = plt.subplots(figsize=(15, 5))
    # Set title and labels
    ax.set_title('Mean reward per episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (mean)')
    # Make scatterplot
    ax.plot(np.arange(1, len(rewards) + 1), np.array(rewards), '.-')
    # Save plot
    plt.savefig(args.agent_path + '/rewards.png')
    # Show plot
    plt.show()
    # Close plot
    plt.close()
