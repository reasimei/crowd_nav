import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib import animation
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.env_config

    # Configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # Configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))

    # Configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # Set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()

    if args.visualize:
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        
        # Collect data for animation
        robot_positions = []
        human_positions = []
        times = []

        # Access the list of humans
        humans = env.humans  # Assuming env.humans is a list of human agents

        while not done:
            action = robot.act(ob)
            ob, _, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            # Get human positions
            human_positions.append([np.array(human.get_position()) for human in humans])
            robot_positions.append(current_pos)
            times.append(env.global_time)
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
            # Optionally, you can render the frame to display it during the simulation
            env.render()

        # Now, create the animation using matplotlib
        fig, ax = plt.subplots()
        # Set plot limits based on the positions collected
        all_robot_positions = np.array(robot_positions)
        all_human_positions = np.array(human_positions)  # Shape: (num_frames, num_humans, 2)
        all_positions = np.concatenate([all_robot_positions, all_human_positions.reshape(-1, 2)])
        min_coords = np.min(all_positions, axis=0)
        max_coords = np.max(all_positions, axis=0)
        ax.set_xlim(min_coords[0] - 1, max_coords[0] + 1)
        ax.set_ylim(min_coords[1] - 1, max_coords[1] + 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Robot and Human Trajectories')

        robot_scatter, = ax.plot([], [], 'ro', label='Robot')
        human_scatters = [ax.plot([], [], 'bo')[0] for _ in range(len(humans))]
        ax.legend()

        def init():
            robot_scatter.set_data([], [])
            for human_scatter in human_scatters:
                human_scatter.set_data([], [])
            return [robot_scatter] + human_scatters

        def animate(i):
            robot_scatter.set_data(robot_positions[i][0], robot_positions[i][1])
            for j, human_scatter in enumerate(human_scatters):
                human_pos = human_positions[i][j]
                human_scatter.set_data(human_pos[0], human_pos[1])
            return [robot_scatter] + human_scatters

        anim = animation.FuncAnimation(fig, animate, frames=len(times), init_func=init, interval=100, blit=True)

        # Save the animation
        if args.traj:
            anim.save('./result/test_020.gif', writer='pillow', fps=10)
            logging.info('Trajectory GIF saved as test_000.gif')
        else:
            # Specify the writer explicitly
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            anim.save('./result/test_006.mp4', writer=writer)
            logging.info('Video saved as test_003.mp4')

        plt.close(fig)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)

    env.close()

if __name__ == '__main__':
    main()
