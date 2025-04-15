import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
import git
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_sim.envs.utils.robot import Robot

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy', type=str, default='sarl')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    # Configure paths and handle output directory
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n): ')
        if key.lower() == 'y' and not args.resume:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
            shutil.copy(args.env_config, args.output_dir)
            shutil.copy(args.policy_config, args.output_dir)
            shutil.copy(args.train_config, args.output_dir)
        else:
            args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
            args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
    else:
        os.makedirs(args.output_dir)
        shutil.copy(args.env_config, args.output_dir)
        shutil.copy(args.policy_config, args.output_dir)
        shutil.copy(args.train_config, args.output_dir)
    log_file = os.path.join(args.output_dir, 'output.log')
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')

    # Configure logging
    mode = 'a' if args.resume else 'w'
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(log_file, mode=mode),
        logging.StreamHandler(sys.stdout)
    ], format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # Configure policy
    policy = policy_factory[args.policy]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)
    policy.set_device(device)

    # Configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    # Initialize robots and set policies
    robot1 = Robot(env_config, 'robot', robot_index=0)
    robot2 = Robot(env_config, 'robot', robot_index=1)
    robot1.set_policy(policy)
    robot2.set_policy(policy)
    env.set_robot(robot1, robot2)

    # Read training parameters
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

    # Configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    batch_size = train_config.getint('trainer', 'batch_size')
    trainer = Trainer(model, memory, device, batch_size)
    explorer = Explorer(env, robot1, robot2, device, memory, policy.gamma, target_policy=policy)

    # Imitation learning
    if args.resume:
        if not os.path.exists(rl_weight_file):
            logging.error('RL weights do not exist')
        else:
            model.load_state_dict(torch.load(rl_weight_file))
            logging.info('Loaded reinforcement learning trained weights. Resuming training.')
    elif os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file))
        logging.info('Loaded imitation learning trained weights.')
    else:
        il_episodes = train_config.getint('imitation_learning', 'il_episodes')
        il_policy_name = train_config.get('imitation_learning', 'il_policy')
        il_epochs = train_config.getint('imitation_learning', 'il_epochs')
        il_learning_rate = train_config.getfloat('imitation_learning', 'il_learning_rate')
        trainer.set_learning_rate(il_learning_rate)
        safety_space = 0 if robot1.visible and robot2.visible else train_config.getfloat('imitation_learning', 'safety_space')

        # Initialize imitation learning policy
        il_policy = policy_factory[il_policy_name]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        robot1.set_policy(il_policy)
        robot2.set_policy(il_policy)

        # Run imitation learning episodes
        explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
        trainer.optimize_epoch(il_epochs)
        torch.save(model.state_dict(), il_weight_file)
        logging.info('Finished imitation learning. Weights saved.')
        logging.info('Experience set size: {}/{}'.format(len(memory), memory.capacity))

    # Update target model
    explorer.update_target_model(model)

    # Reinforcement learning setup
    policy.set_env(env)
    robot1.set_policy(policy)
    robot2.set_policy(policy)
    robot1.print_info()
    robot2.print_info()
    trainer.set_learning_rate(rl_learning_rate)

    # Fill the memory pool with some RL experience if resuming
    if args.resume:
        robot1.policy.set_epsilon(epsilon_end)
        robot2.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
        logging.info('Experience set size: {}/{}'.format(len(memory), memory.capacity))

    episode = 0
    while episode < train_episodes:
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot1.policy.set_epsilon(epsilon)
        robot2.policy.set_epsilon(epsilon)

        # Evaluate the model
        if episode % evaluation_interval == 0 and episode != 0:
            explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)

        # Sample episodes and optimize the model
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        trainer.optimize_batch(train_batches)
        episode += 1

        # Update the target model
        if episode % target_update_interval == 0:
            explorer.update_target_model(model)

        # Save checkpoint
        if episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), rl_weight_file)
            logging.info('Saved checkpoint at episode {}'.format(episode))

    # Final test after training
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode)
    logging.info('Training completed. Models saved in {}'.format(args.output_dir))

if __name__ == '__main__':
    main()