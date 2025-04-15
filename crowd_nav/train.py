import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
import git
import numpy as np
import math
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.parser import args, parser

def main():
    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y' and not args.resume:
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
            args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.env_config, args.output_dir)
        shutil.copy(args.policy_config, args.output_dir)
        shutil.copy(args.train_config, args.output_dir)
    log_file = os.path.join(args.output_dir, 'output.log')
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')

    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    logging.info('Current git head hash code: %s'.format(repo.head.object.hexsha))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    if args.policy_config is None:
        parser.error('Policy config has to be specified for a trainable network')
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)
    policy.set_device(device)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    # robot1 = Robot(env_config, 'robot', robot_index=0)
    # robot2 = Robot(env_config, 'robot', robot_index=1)
    # robot = [robot1, robot2]
    # env.set_robot(robot1, robot2)
    robot_num = env_config.getint('sim', 'robot_num')
    robots=[None] * robot_num
    for i in range(robot_num):
        robots[i] = Robot(env_config, 'robot', robot_index=i)
    env.set_robots(robots)    


    # read training parameters
    if args.train_config is None:
        parser.error('Train config has to be specified for a trainable network')
    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    rl_learning_rate = train_config.getfloat('train', 'rl_learning_rate')
    train_batches = train_config.getint('train', 'train_batches')
    train_episodes = train_config.getint('train', 'train_episodes')
    sample_episodes = train_config.getint('train', 'sample_episodes')
    target_update_interval = train_config.getint('train', 'target_update_interval')
    evaluation_interval = train_config.getint('train', 'evaluation_interval')
    capacity = train_config.getint('train', 'capacity')
    epsilon_start = train_config.getfloat('train', 'epsilon_start')
    epsilon_end = train_config.getfloat('train', 'epsilon_end')
    epsilon_decay = train_config.getfloat('train', 'epsilon_decay')
    checkpoint_interval = train_config.getint('train', 'checkpoint_interval')

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    batch_size = train_config.getint('trainer', 'batch_size')
    trainer = Trainer(model, memory, device, batch_size)
    # explorer = Explorer(env, robot1, robot2, device, memory, policy.gamma, target_policy=policy)
    explorer = Explorer(env, robots, device, memory, policy.gamma, target_policy=policy)
    # imitation learning
    if args.resume:
        if os.path.exists(rl_weight_file):
            model.load_state_dict(torch.load(rl_weight_file))
            logging.info('Load reinforcement learning trained weights. Resume training')
        elif os.path.exists(il_weight_file):
            model.load_state_dict(torch.load(il_weight_file))
            logging.info('Load imitation learning trained weights. Start RL training')
        else:
            logging.error('Neither RL nor IL weights exist')
            return
    elif os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file))
        logging.info('Load imitation learning trained weights.')
    else:
        il_episodes = train_config.getint('imitation_learning', 'il_episodes')
        il_policy = train_config.get('imitation_learning', 'il_policy')
        il_epochs = train_config.getint('imitation_learning', 'il_epochs')
        il_learning_rate = train_config.getfloat('imitation_learning', 'il_learning_rate')
        trainer.set_learning_rate(il_learning_rate)
        for robot in robots:
            if robot.visible:
                safety_space = 0
            else:
                safety_space = train_config.getfloat('imitation_learning', 'safety_space')
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        for i in range(robot_num):
            robots[i].set_policy(il_policy)
            
        explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
        trainer.optimize_epoch(il_epochs)
        torch.save(model.state_dict(), il_weight_file)
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    explorer.update_target_model(model)

    # reinforcement learning
    policy.set_env(env)
    # robot1.set_policy(policy)
    # robot2.set_policy(policy)
    # robot1.print_info()
    # robot2.print_info()
    for i in range(robot_num):
        robots[i].set_policy(policy)
        robots[i].print_info()
    trainer.set_learning_rate(rl_learning_rate)
    # fill the memory pool with some RL experience
    if args.resume:
        for robot in robots:
            robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    # After explorer.update_target_model(model) and before the training loop starts
    #torch.save(model.state_dict(), rl_weight_file)  # Save initial model at episode 0
    episode = 0
    while episode < train_episodes:
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        for robot in robots:
            robot.policy.set_epsilon(epsilon)
        # robot1.policy.set_epsilon(epsilon)
        # robot2.policy.set_epsilon(epsilon)

        # evaluate the model
        if episode % evaluation_interval == 0:
            explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        trainer.optimize_batch(train_batches)
        episode += 1

        if episode % target_update_interval == 0:
            explorer.update_target_model(model)

        if episode != 0 and episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), rl_weight_file)
        
        losses = trainer.losses
        optimizer = trainer.optimizer

        # 记录每个batch的平均损失
        avg_loss = np.mean(losses)
        if avg_loss == 0 or math.isnan(avg_loss):
            logging.warning(f"Zero or NaN loss detected in batch {episode}! Check reward calculation.")

        # 增加学习率热启动和衰减
        if hasattr(optimizer, 'param_groups'):
            # 前20个epoch使用热启动
            if episode < 20:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = min(1e-3, 1e-4 * (1 + episode/10))
            # 之后使用衰减
            elif episode % 50 == 0 and episode > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95

        # Reset training if consecutive NaN/zero losses detected
        zero_loss_counter = 0
        for loss in losses[-5:]:
            if loss == 0 or math.isnan(loss):
                zero_loss_counter += 1

        if zero_loss_counter >= 3:
            logging.warning("Multiple zero/NaN losses detected. Resetting optimizer.")
            trainer.set_learning_rate(rl_learning_rate * 0.5)  # Try lower learning rate
            # Clear some of the memory to remove problematic experiences
            if len(memory) > 1000:
                memory.memory = memory.memory[:1000]

    # final test
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode)


if __name__ == '__main__':
    main()
