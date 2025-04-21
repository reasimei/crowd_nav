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
from crowd_sim.envs.utils.action import ActionXY, ActionRot


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
    parser.add_argument('--hose', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true', help='Enable debug mode with more logging')
    parser.add_argument('--pause', default=False, action='store_true', help='Pause between steps to observe robot behavior')
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
        try:
            # 详细记录模型加载信息
            logging.info(f"Attempting to load model from: {model_weights}")
            policy.get_model().load_state_dict(torch.load(model_weights, map_location=device))
            logging.info(f"Model weights loaded successfully")
            # 添加模型信息记录
            logging.info(f"Policy type: {type(policy).__name__}")
            logging.info(f"Model type: {type(policy.get_model()).__name__}")
        except Exception as e:
            logging.error(f"Failed to load model weights: {e}")
            logging.error(f"Will continue with untrained model")

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
    phase_changed = False
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
        is_lead = [None]*robot_num
        lead = [None]*robot_num
        follow = [None]*robot_num

        for i in range(robot_num):
            partner = robots[i].get_hose_partner()
            if partner is not None:
                lead[i],follow[i],is_lead[i] = robots[i].identify_lead_follow_roles()
        # Access the list of humans
        humans = env.humans  # Assuming env.humans is a list of human agents

        while not done:
            # 人类避障/机器人协调训练阶段切换
            if (args.policy == 'h_sarl' or args.policy == 'h_llm_sarl'):
                # 如果所有人类都到达目标，切换到机器人协调阶段
                if not phase_changed and all(h.reached_destination() for h in humans):
                   # 传递新阶段给所有机器人
                    for robot in robots:
                        robot.training_phase = 'robot_avoidance'
                        phase_changed = True


            # Each robot uses its own observation
            flag = 0
            actions = []
            current_pos = []
            for i in range(robot_num):
                try:
                    robot = robots[i]
                    if args.policy == 'h_sarl' or args.policy == 'h_llm_sarl':
                        # Use the same specialized methods as in training
                        if robot.training_phase == 'human_avoidance':
                            print('training_phase:human_avoidance')
                            robot_action = robot.act_avoid_humans(ob)
                        elif robot.training_phase == 'robot_avoidance':
                            print('training_phase:robot_avoidance')
                            robot_action,flag = robot.act_avoid_robots(ob,flag,lead[i],is_lead[i])
                        else:
                            robot_action = robot.act(ob)
                    else:
                        # For non-hierarchical policies, use the standard act method
                        robot_action = robot.act(ob)
                    
                    # Ensure action is valid
                    if robot_action is None or not (hasattr(robot_action, 'vx') and hasattr(robot_action, 'vy')):
                        logging.warning(f"Robot {i} action is None or not ActionXY/ActionRot,return stop action")
                        if robot.kinematics == 'holonomic':
                            robot_action = ActionXY(0, 0)
                        else:
                            robot_action = ActionRot(0, 0)
                        
                    actions.append(robot_action)
                except Exception as e:
                    logging.error(f"Error in robot {i} action selection: {e}")
                    # Provide safe default action
                    if robots[i].kinematics == 'holonomic':
                        actions.append(ActionXY(0, 0))
                    else:
                        actions.append(ActionRot(0, 0))

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
            # Optionally, you can render the frame to display it during the simulation
            env.render()
            # 在无头环境中，不使用 plt.pause()
            #fig.canvas.draw()
            #fig.canvas.flush_events()
            if args.pause:
                try:
                    input("Press Enter to continue to next step...")
                except KeyboardInterrupt:
                    done = True
        # 在程序结束时清理
        # if args.visualize:
        #     plt.close('all')

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

        # Initialize the hose line (or empty list if hose not used)
        if args.hose:
            hose_lines = []
            for i in range(robot_num//2):
                line, = ax.plot([], [], 'k-', linewidth=2, label='Hose')
                hose_lines.append(line)
        else:
            hose_lines = []  # 创建空列表

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
            # 清理可能存在的旧图形对象
            plt.close('all')
            
            for robot_circle in robot_circles:
                robot_circle.center = (0, 0)
            for human_circle in human_circles:
                human_circle.center = (0, 0)
            if args.hose:
                for hose_line in hose_lines:
                    hose_line.set_data([], [])
            return robot_circles + human_circles + hose_lines

        def animate(frame):
            for i, robot_circle in enumerate(robot_circles):
                robot_pos = robot_positions[frame][i]
                robot_circle.center = (robot_pos[0], robot_pos[1])
            for j, human_circle in enumerate(human_circles):
                human_pos = human_positions[frame][j]
                human_circle.center = (human_pos[0], human_pos[1])
            
            # 软管处理 - 只在启用软管时执行
            if args.hose:
                hose_length = 2.0
                robot_pos = np.array([]).reshape(0, 2)
                for i in range(robot_num):
                    robot_pos = np.vstack([robot_pos, [robot_positions[frame][i][0], robot_positions[frame][i][1]]])
                
                x = [None] * (int)(robot_num/2)
                y = [None] * (int)(robot_num/2)
                
                for i in range((int)(robot_num/2)):
                    x[i], y[i] = hose_model(robot_pos[i*2], robot_pos[i*2+1], hose_length)
                    hose_lines[i].set_data(x[i], y[i])

            return robot_circles + human_circles + hose_lines

        anim = animation.FuncAnimation(fig, animate, frames=len(times), init_func=init,
                                    interval=100, blit=True)

        # Save the animation
        try:
            if args.traj:
                anim.save('./result/hllmsarl/test_038_2.gif', writer='pillow', fps=10)
                logging.info('Trajectory GIF saved as test_001.gif')
            else:
                # 检查 ffmpeg 是否可用
                try:
                    Writer = animation.writers['ffmpeg']
                    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                    anim.save('./result/test_000.mp4', writer=writer)
                    logging.info('Video saved as test_000.mp4')
                except RuntimeError:
                    logging.warning('ffmpeg not available, saving as GIF instead')
                    anim.save('./result/test_fallback.gif', writer='pillow', fps=10)
                    logging.info('Fallback GIF saved as test_fallback.gif')
        except Exception as e:
            logging.error(f"Error saving animation: {e}")
            plt.savefig('./result/last_frame.png')
            logging.info('Saved last frame as PNG instead')
            
        # 确保所有图形都被关闭
        plt.close('all')

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
