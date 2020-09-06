# Dependencies
from src.environment import Environment
from src.agent import Agent
import numpy as np
import argparse
import dill


# Main
if __name__ == '__name__':

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
        '--start_point', type=int, nargs='+', required=False,
        help='Define a starting point for the agent'
    )
    # Add argument: goal point
    parser.add_argument(
        '--goal_point', type=int, nargs='+', required=False,
        help='Define a goal point to reach for the agent'
    )
    # Add argument: discount percentage
    parser.add_argument(
        '--prc_discount', type=float, default=0.9,
        help='Discount percentage to apply'
    )
    # Add argument: whether to use softmax or not
    parser.add_argument(
        '--use_softmax', type=int, default=1,
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
    # Add argument: save to defined file
    parser.add_argumen(
        '--agent_path', type=str, required=False,
        help='Path where to save trained model'
    )
    # Add argument: verbose log
    parser.add_argument(
        '--verbose', '-v', type=int, default=1,
        help='Whether to print out verbose log or not'
    )
    # Parse arguments
    args = parser.parse_args()

    # Define available actions
    actions = [*Environment.actions.values()]

    # Define alpha factors
    alpha = np.ones(args.num_episodes, dtype=np.float) * 0.25
    # Define epsilon factor
    epsilon = np.linspace(0.8, 0.001, args.num_episodes, dtype=np.float)

    # Initialize the agent
    agent = Agent(
        num_states=(args.env_width * args.env_height),
        num_actions=len(actions)
    )

    # Sample goal point at random
    x = np.random.randint(args.env_width)
    y = np.random.randint(args.env_height)
    # Define new goal point
    goal_point = np.array([x, y], dtype=np.int)

    # Case goal point has been defined by the user
    if args.goal_point is not None:
        # Use user defined goal point
        goal_point = np.array(args.goal_point, dtype=np.int)

    # Loop through each training episode
    for episode in range(args.num_episodes):

        # Initialize reward
        reward = 0.0

        # Sample x and y at random
        x = np.random.randint(args.env_width)
        y = np.random.randint(args.env_height)
        # Define starting point
        start_point = np.array([x, y], dtype=np.int)

        # Case starting point has not been defined by the user
        if args.start_point is not None:
            # Use user defined starting point
            start_point = np.array(args.start_point, dtype=np.int)

        # Define current state as starting point
        state = start_point
        # Define new environment
        env = Environment(
            width=args.env_width,
            height=args.env_height,
            start=start_point,
            goal=goal_point
        )

        # Loop through each step in current training episode
        for step in range(args.len_episodes):

            # Define current cell (flattened) index
            _state = state[0] * args.env_height + state[1]
            # Choose an action
            action = agent.choose_action(_state, epsilon[episode])
            # Retrieve next state and reward
            next_state, _reward = env.move(action)
            # Flatten next state index
            _next_state = next_state[0] * args.env_height + next_state[1]
            # Q-learning update
            agent.update_state(_state, action, _reward, _next_state, alpha[episode], epsilon[episode])
            # Update state
            state = next_state
            # Update reward
            reward += _reward

        # Update reward (compute average)
        reward /= args.len_episodes

        # Save the agent
        if ((episode + 1) % args.save_after == 0):
            # Open output file
            with open(args.agent_path, 'wb') as agent_file:
                # Write out to file
                dill.dump(agent, agent_file)

        # Verbose log
        if args.verbose:
            # Define output message
            msg = 'Episode %d:' % (episode + 1)
            msg += 'the agent has obtained an average reward '
            msg += 'of %.03f ' % reward
            msg += 'starting from position %d, %d' % tuple(start_point)
            # Print output message
            print(msg)
