#!/usr/bin/env python3
import rospy
import rospkg
import argparse
import configparser
import os
import sys
import logging
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

# ROS Imports
from geometry_msgs.msg import Twist

def tuple_to_twist(action_tuple):
    twist = Twist()
    twist.linear.x = action_tuple[0]
    twist.linear.y = action_tuple[1]
    twist.linear.z = action_tuple[2] if len(action_tuple) > 2 else 0.0
    twist.angular.x = action_tuple[3] if len(action_tuple) > 3 else 0.0
    twist.angular.y = action_tuple[4] if len(action_tuple) > 4 else 0.0
    twist.angular.z = action_tuple[5] if len(action_tuple) > 5 else 0.0
    return twist

def main():
    # Initialize ROS node
    rospy.init_node('upper_computer_node', anonymous=True)

    # Initialize ROS Publishers for robot1 and robot2 cmd_vel
    cmd_vel_pub_robot1 = rospy.Publisher('/robot_1/cmd_vel', Twist, queue_size=10)
    cmd_vel_pub_robot2 = rospy.Publisher('/robot_2/cmd_vel', Twist, queue_size=10)

    # Argument Parsing
    parser = argparse.ArgumentParser(description='Parse configuration file')
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
    
    # Use parse_known_args to ignore unknown ROS arguments
    args, unknown = parser.parse_known_args()

    # Handle model weights path
    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            resumed_rl_model = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            rl_model = os.path.join(args.model_dir, 'rl_model.pth')
            if os.path.exists(resumed_rl_model):
                model_weights = resumed_rl_model
            else:
                model_weights = rl_model
    else:
        env_config_file = args.env_config
        policy_config_file = args.policy_config  # Fixed to read policy_config instead of env_config
        model_weights = None  # Ensure model_weights is initialized

    # Configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # Ensure the 'result' directory exists
    if not os.path.exists('./result'):
        os.makedirs('./result')

    # Configure policy
    policy = policy_factory.get(args.policy)
    if policy is None:
        logging.error(f"Policy '{args.policy}' not found in policy_factory.")
        sys.exit(1)
    policy = policy()
    policy_config = configparser.RawConfigParser()
    if not os.path.exists(policy_config_file):
        logging.error(f"Policy configuration file not found at {policy_config_file}")
        sys.exit(1)
    policy_config.read(policy_config_file)
    if not policy_config.has_section('rl'):
        logging.error("The policy configuration file is missing the [rl] section.")
        sys.exit(1)
    policy.configure(policy_config)
    if hasattr(policy, 'trainable') and policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        if not os.path.exists(model_weights):
            logging.error(f"Model weights file not found at {model_weights}")
            sys.exit(1)
        policy.get_model().load_state_dict(torch.load(model_weights, map_location=device))

    # Configure environment
    env_config = configparser.RawConfigParser()
    if not os.path.exists(env_config_file):
        logging.error(f"Environment configuration file not found at {env_config_file}")
        sys.exit(1)
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot1 = Robot(env_config, 'robot', robot_index=1)
    robot2 = Robot(env_config, 'robot', robot_index=2)
    robots = [robot1, robot2]
    robot1.set_policy(policy)
    robot2.set_policy(policy)
    env.set_robot(robot1, robot2)
    explorer = Explorer(env, robot1, robot2, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # Set safety space for ORCA in non-cooperative simulation
    if isinstance(robot1.policy, ORCA) and isinstance(robot2.policy, ORCA):
        robot1.policy.safety_space = 0
        robot2.policy.safety_space = 0
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

            # Publish the generated velocities to ROS topics
            twist_robot1 = tuple_to_twist(action1)
            cmd_vel_pub_robot1.publish(twist_robot1)
            logging.info(f'Published cmd_vel for Robot 1: {twist_robot1}')

            twist_robot2 = tuple_to_twist(action2)
            cmd_vel_pub_robot2.publish(twist_robot2)
            logging.info(f'Published cmd_vel for Robot 2: {twist_robot2}')

        # Now, create the animation using matplotlib
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
        if human_positions.size > 0:
            all_human_positions_flat = human_positions.reshape(-1, 2)
            all_positions = np.concatenate([all_robot_positions_flat, all_human_positions_flat])
        else:
            all_positions = all_robot_positions_flat

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
            circle = Circle((0, 0), 0.2, color=robot_colors[i], zorder=5)
            ax.add_patch(circle)
            robot_circles.append(circle)

        # Define human colors
        human_colors = plt.cm.get_cmap('tab20').colors
        human_colors = human_colors[:len(humans)] if len(humans) <= len(human_colors) else ['orange'] * len(humans)

        # Initialize circle patches for humans
        human_circles = []
        for i in range(len(humans)):
            circle = Circle((0, 0), 0.1, color=human_colors[i], zorder=5)
            ax.add_patch(circle)
            human_circles.append(circle)

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
            for circle in robot_circles:
                circle.center = (-100, -100)  # Initialize off-screen
            for circle in human_circles:
                circle.center = (-100, -100)  # Initialize off-screen
            return robot_circles + human_circles

        def animate(frame):
            for i, circle in enumerate(robot_circles):
                circle.center = robot_positions[frame, i]
            for i, circle in enumerate(human_circles):
                if frame < len(human_positions):
                    circle.center = human_positions[frame, i]
                else:
                    circle.center = (-100, -100)  # Hide if no data
            return robot_circles + human_circles

        anim = animation.FuncAnimation(fig, animate, frames=len(times), init_func=init,
                                       interval=100, blit=True)

        # Save the animation
        if args.traj:
            anim.save(args.video_file if args.video_file else 'trajectory.mp4', writer='ffmpeg')
        else:
            anim.save('trajectory.gif', writer='imagemagick')

        plt.close(fig)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot1.visible and robot2.visible and info == 'reach goal':
            logging.info('Both robots have reached their goals successfully.')

    else:
        # Run reinforcement learning episodes and publish cmds
        for episode in range(env.case_size[args.phase]):
            ob = env.reset(args.phase, args.test_case)
            done = False
            while not done:
                # Each robot uses its own observation
                action1 = robot1.act(ob)
                action2 = robot2.act(ob)
                ob, _, done, info = env.step(action1, action2)

                # Create Twist messages for each robot
                twist_robot1 = tuple_to_twist(action1)
                twist_robot2 = tuple_to_twist(action2)

                # Publish the Twist messages to respective topics
                cmd_vel_pub_robot1.publish(twist_robot1)
                cmd_vel_pub_robot2.publish(twist_robot2)

                logging.info(f'Published cmd_vel for Robot 1: {twist_robot1}')
                logging.info(f'Published cmd_vel for Robot 2: {twist_robot2}')

                # Optional: Add a sleep rate if necessary
                rospy.sleep(0.1)  # Sleep for 100ms

    env.close()

if __name__ == '__main__':
    import sys
    import logging
    import torch
    import configparser
    import argparse
    import rospy

    main()