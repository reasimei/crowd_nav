#!/usr/bin/env python3
import rospy
import rospkg
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
from crowd_nav.msg import PedestrianPositions

# ROS Imports
from geometry_msgs.msg import Point
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

    # 使用 parse_known_args 而不是 parse_args
    args, unknown = parser.parse_known_args()

    # 从 ROS 参数服务器获取参数
    args.policy = rospy.get_param('~policy', args.policy)
    args.model_dir = rospy.get_param('~model_dir', args.model_dir)
    args.phase = rospy.get_param('~phase', args.phase)
    args.test_case = rospy.get_param('~test_case', args.test_case)
    args.visualize = rospy.get_param('~visualize', args.visualize)
    args.traj = rospy.get_param('~traj', args.traj)

    # 获取机器人数量
    robot_num = rospy.get_param('/robot_num', 2)  # 默认值为2
    rospy.loginfo(f"Initializing with {robot_num} robots")

    # Initialize ROS Publishers for all robots' cmd_vel
    cmd_vel_publishers = {}
    for i in range(robot_num):
        topic_name = f'/robot_{i+1}/cmd_vel'
        cmd_vel_publishers[i] = rospy.Publisher(topic_name, Twist, queue_size=10)
        rospy.loginfo(f"Created publisher for {topic_name}")


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

    # 确保使用正确的配置文件路径
    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        rospy.loginfo(f"Using env config file: {env_config_file}")
        
        if not os.path.exists(env_config_file):
            rospy.logerr(f"Environment config file not found: {env_config_file}")
            return
    
    # 读取并验证配置
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    
    # 验证必要的配置部分是否存在
    required_sections = ['env', 'reward', 'sim', 'humans', 'robot', 'hose']
    for section in required_sections:
        if not env_config.has_section(section):
            rospy.logerr(f"Missing required section '{section}' in config file")
            return

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
        policy.get_model().load_state_dict(torch.load(model_weights, map_location=device))

    # Configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    
    # expand robot numbers
    robot_num = env_config.getint('sim', 'robot_num')
    robots=[None] * robot_num
    for i in range(robot_num):
        robots[i] = Robot(env_config, 'robot', robot_index=i)
        robots[i].set_policy(policy)
    env.set_robots(robots)    
    explorer = Explorer(env, robots, device, gamma=0.9)
    
    # robot1 = Robot(env_config, 'robot', robot_index=0)
    # robot2 = Robot(env_config, 'robot', robot_index=1)
    # robot = [robot1, robot2]
    # robot1.set_policy(policy)
    # robot2.set_policy(policy)
    # env.set_robot(robot1, robot2)
    # explorer = Explorer(env, robot1, robot2, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # Set safety space for ORCA in non-cooperative simulation
    # if isinstance(robot1.policy, ORCA) and isinstance(robot2.policy, ORCA):
    #     if robot1.visible and robot2.visible:
    #         robot1.policy.safety_space = 0
    #     else:
    #         robot1.policy.safety_space = 0
    #     logging.info('ORCA agent buffer: %f', robot1.policy.safety_space)
    for i in range(robot_num):
        robots[i].policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robots[i].policy.safety_space)

    policy.set_env(env)
    for i in range(robot_num):
        robots[i].print_info()
    # robot1.print_info()
    # robot2.print_info()

    if args.visualize:
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = []
        for i in range(robot_num):
            last_pos.append(np.array(robots[i].get_position()))
        # last_pos1 = np.array(robot1.get_position())
        # last_pos2 = np.array(robot2.get_position())
        # last_pos = [last_pos1, last_pos2]
        
        # Collect data for animation
        robot_positions = []
        human_positions = []
        times = []

        # Access the list of humans
        humans = env.humans  # Assuming env.humans is a list of human agents

        while not done:
            # Each robot uses its own observation
            actions = []
            current_pos = []
            for i in range(robot_num):
                actions.append(robots[i].act(ob)) 
            # action1 = robot1.act(ob)
            # action2 = robot2.act(ob)
            # ob, _, done, info = env.step(action1, action2)
            ob, _, done, info = env.step1(actions)
            for i in range(robot_num):
                current_pos.append(np.array(robots[i].get_position())) 
            # current_pos1 = np.array(robot1.get_position())
            # current_pos2 = np.array(robot2.get_position())
            # current_pos = [current_pos1, current_pos2]
            # Get human positions
            human_positions.append([np.array(human.get_position()) for human in humans])
            robot_positions.append(current_pos)
            times.append(env.global_time)
            for i in range(robot_num):
                logging.debug('Robot%d Speed: %.2f', i+1, np.linalg.norm(current_pos[i] - last_pos[i]) / robots[i].time_step)
            # logging.debug('Robot1 Speed: %.2f', np.linalg.norm(current_pos1 - last_pos1) / robot1.time_step)
            # logging.debug('Robot2 Speed: %.2f', np.linalg.norm(current_pos2 - last_pos2) / robot2.time_step)
            # last_pos1 = current_pos1
            # last_pos2 = current_pos2
            last_pos = current_pos

            # Publish the action to the robot
            # Publish the generated velocities to ROS topics for each robot
            for i, action in enumerate(actions):
                twist = tuple_to_twist(action)
                cmd_vel_publishers[i].publish(twist)
                logging.info(f'Published cmd_vel for Robot {i+1}: {twist}')
            
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
        base_colors = ['blue', 'cyan', 'green', 'yellow', 'gray', 'purple', 'brown', 'lightgreen']
        robot_colors = []

        # Generate colors for all robots by cycling through base colors
        for i in range(robot_num):
            robot_colors.append(base_colors[i % len(base_colors)])

        # Initialize circle patches for robots
        robot_circles = []
        for i in range(robot_num):
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
        hose_lines = [Line2D] * (robot_num//2)
        for i in range(robot_num//2):
            hose_lines[i], = ax.plot([], [], 'k-', linewidth=2, label='Hose')
        # hose_line, = ax.plot([], [], 'k-', linewidth=2, label='Hose')

        # Create custom legend handles
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Robot {i+1}',
                                markerfacecolor=robot_colors[i], markersize=10,
                                markeredgecolor='black', markeredgewidth=1.0)
                        for i in range(robot_num)] + \
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
            for hose_line in hose_lines:
                 hose_line.set_data([], [])
            # hose_line.set_data([], [])
            return robot_circles + human_circles + hose_lines

        def animate(frame):
            for i, robot_circle in enumerate(robot_circles):
                robot_pos = robot_positions[frame][i]
                robot_circle.center = (robot_pos[0], robot_pos[1])
            for j, human_circle in enumerate(human_circles):
                human_pos = human_positions[frame][j]
                human_circle.center = (human_pos[0], human_pos[1])
            # print("Frame:", frame)
            # print("robot_positions[0][0][0]=",robot_positions[0][0][0])
            hose_length = 2.0
            robot_pos = np.array([]).reshape(0, 2)
            for i in range(robot_num):
                robot_pos = np.vstack([robot_pos, [robot_positions[frame][i][0], robot_positions[frame][i][1]]])
            # robot1_pos = [robot_positions[frame][0][0], robot_positions[frame][0][1]]
            # robot2_pos = [robot_positions[frame][1][0], robot_positions[frame][1][1]]
            # print("robot_pos=",robot_pos)
            x = [None] * (int)(robot_num/2)
            y = [None] * (int)(robot_num/2)
            for i in range((int)(robot_num/2)):
                # print("robot_pos[0],[1]=",robot_pos[0],' ',robot_pos[1])
                x[i],y[i] = hose_model(robot_pos[i*2], robot_pos[i*2+1], hose_length)
                hose_lines[i].set_data(x[i],y[i])

            
            # Update the hose line between the two robots
            # xdata = [robot_positions[frame][0][0], robot_positions[frame][1][0]]
            # ydata = [robot_positions[frame][0][1], robot_positions[frame][1][1]]
            # hose_line.set_data(xdata, ydata)
            


            # return robot_circles + human_circles + [hose_line]
            return robot_circles + human_circles + hose_lines

        anim = animation.FuncAnimation(fig, animate, frames=len(times), init_func=init,
                                    interval=100, blit=True)

        # Save the animation
        if args.traj:
            anim.save('./result/test_001.gif', writer='pillow', fps=10)
            logging.info('Trajectory GIF saved as test_001.gif')
        else:
            # Specify the writer explicitly
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            anim.save('./result/test_000.mp4', writer=writer)
            logging.info('Video saved as test_000.mp4')

        plt.close(fig)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if all(robot.visible for robot in robots) and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)

    env.close()


def set_robots(self, robots):
    self.robots = robots
    for robot in self.robots:
        robot.set_env(self)


if __name__ == '__main__':
    main()
