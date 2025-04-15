import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from matplotlib.lines import Line2D
from crowd_sim.envs.utils.utils import hose_model
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
    robot1 = Robot(env_config, 'robot', robot_index=0)
    robot2 = Robot(env_config, 'robot', robot_index=1)
    robot = [robot1, robot2]
    robot1.set_policy(policy)
    robot2.set_policy(policy)
    env.set_robot(robot1, robot2)
    explorer = Explorer(env, robot1, robot2, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # Set safety space for ORCA in non-cooperative simulation
    if isinstance(robot1.policy, ORCA) and isinstance(robot2.policy, ORCA):
        if robot1.visible and robot2.visible:
            robot1.policy.safety_space = 0
        else:
            robot1.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot1.policy.safety_space)

    policy.set_env(env)
    robot1.print_info()
    robot2.print_info()

    if args.visualize:
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos1 = np.array(robot1.get_position())
        last_pos2 = np.array(robot2.get_position())
        last_pos = [last_pos1, last_pos2]
        
        # Collect data for animation
        robot_positions = []
        human_positions = []
        times = []

        # Access the list of humans
        humans = env.humans  # Assuming env.humans is a list of human agents

        while not done:
            # Each robot uses its own observation
            action1 = robot1.act(ob)
            action2 = robot2.act(ob)
            ob, _, done, info = env.step(action1, action2)
            current_pos1 = np.array(robot1.get_position())
            current_pos2 = np.array(robot2.get_position())
            current_pos = [current_pos1, current_pos2]
            # Get human positions
            human_positions.append([np.array(human.get_position()) for human in humans])
            robot_positions.append(current_pos)
            times.append(env.global_time)
            logging.debug('Robot1 Speed: %.2f', np.linalg.norm(current_pos1 - last_pos1) / robot1.time_step)
            logging.debug('Robot2 Speed: %.2f', np.linalg.norm(current_pos2 - last_pos2) / robot2.time_step)
            last_pos1 = current_pos1
            last_pos2 = current_pos2
            last_pos = current_pos
            # Optionally, you can render the frame to display it during the simulation
            env.render()

        # Now, create the animation using matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
        from matplotlib import animation

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.tick_params(labelsize=16)
        ax.set_xlabel('x (m)', fontsize=16)
        ax.set_ylabel('y (m)', fontsize=16)
        ax.set_title('Robots and Human Trajectories', fontsize=18)

        # Convert lists to numpy arrays
        robot_positions = np.array(robot_positions)  # Shape: (num_frames, 2, 2)
        human_positions = np.array(human_positions)  # Shape: (num_frames, num_humans, 2)

        # Concatenate positions for plot limits
        all_robot_positions_flat = robot_positions.reshape(-1, 2)
        all_human_positions_flat = human_positions.reshape(-1, 2)
        all_positions = np.concatenate([all_robot_positions_flat, all_human_positions_flat])

        # Calculate plot limits
        min_coords = np.min(all_positions, axis=0)
        max_coords = np.max(all_positions, axis=0)

        # Set plot limits based on the positions collected
        ax.set_xlim(min_coords[0] - 1, max_coords[0] + 1)
        ax.set_ylim(min_coords[1] - 1, max_coords[1] + 1)
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Define robot colors matching the static diagram
        robot_colors = ['blue', 'cyan']  # 'blue' for robot1, 'cyan' for robot2

        # Initialize circle patches for robots
        robot_circles = []
        for i in range(2):
            robot_circle = Circle((0, 0), 0.3, facecolor=robot_colors[i], edgecolor='black',
                                linewidth=1.0, zorder=3, label=f'Robot {i+1}')
            ax.add_patch(robot_circle)
            robot_circles.append(robot_circle)

        # Define human colors
        human_colors_dict = {
            0: 'orange',  # human1
            2: 'pink',    # human3
            3: 'red'      # human4
            # Assign colors to other humans as needed
        }
        default_colors = plt.cm.get_cmap('tab20').colors
        human_base_colors = []
        for i in range(len(humans)):
            if i in human_colors_dict:
                base_color = human_colors_dict[i]
            else:
                base_color = default_colors[i % len(default_colors)]
            human_base_colors.append(base_color)
        human_colors = human_base_colors

        # Initialize circle patches for humans
        human_circles = []
        for i in range(len(humans)):
            human_circle = Circle((0, 0), 0.3, facecolor=human_colors[i], edgecolor='black',
                                linewidth=1.0, zorder=3, label=f'Human {i+1}')
            ax.add_patch(human_circle)
            human_circles.append(human_circle)

        # Initialize the hose line
        hose_line, = ax.plot([], [], 'k-', linewidth=2, label='Hose')

        # Create custom legend handles
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Robot {i+1}',
                                markerfacecolor=robot_colors[i], markersize=10,
                                markeredgecolor='black', markeredgewidth=1.0)
                        for i in range(2)] + \
                        [Line2D([0], [0], marker='o', color='w', label=f'Human {i+1}',
                                markerfacecolor=human_colors[i], markersize=10,
                                markeredgecolor='black', markeredgewidth=1.0)
                        for i in range(len(humans))]
        ax.legend(handles=legend_elements, fontsize=12, loc='upper right')

        def init():
            for robot_circle in robot_circles:
                robot_circle.center = (0, 0)
            for human_circle in human_circles:
                human_circle.center = (0, 0)
            hose_line.set_data([], [])
            return robot_circles + human_circles + [hose_line]

        def animate(frame):
            for i, robot_circle in enumerate(robot_circles):
                robot_pos = robot_positions[frame][i]
                robot_circle.center = (robot_pos[0], robot_pos[1])
            for j, human_circle in enumerate(human_circles):
                human_pos = human_positions[frame][j]
                human_circle.center = (human_pos[0], human_pos[1])
            
            robot1_pos = [robot_positions[frame][0][0], robot_positions[frame][0][1]]
            robot2_pos = [robot_positions[frame][1][0], robot_positions[frame][1][1]]
            hose_length = 2.0
            xdata, ydata = hose_model(robot1_pos, robot2_pos, hose_length)
            hose_line.set_data(xdata, ydata)
            
            # Update the hose line between the two robots
            # xdata = [robot_positions[frame][0][0], robot_positions[frame][1][0]]
            # ydata = [robot_positions[frame][0][1], robot_positions[frame][1][1]]
            # hose_line.set_data(xdata, ydata)
            


            return robot_circles + human_circles + [hose_line]

        anim = animation.FuncAnimation(fig, animate, frames=len(times), init_func=init,
                                    interval=100, blit=True)

        # Save the animation
        if args.traj:
            anim.save('./result/test_002.gif', writer='pillow', fps=10)
            logging.info('Trajectory GIF saved as test_000.gif')
        else:
            # Specify the writer explicitly
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            anim.save('./result/test_000.mp4', writer=writer)
            logging.info('Video saved as test_000.mp4')

        plt.close(fig)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot1.visible and robot2.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)

    env.close()

if __name__ == '__main__':
    main()
