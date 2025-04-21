import re
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



def running_mean(x, n):
    if len(x) < n:
        n = len(x)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    result = (cumsum[n:] - cumsum[:-n]) / float(n)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+')
    parser.add_argument('--plot_reward', default=True, action='store_true')
    parser.add_argument('--plot_train', default=True, action='store_true')
    parser.add_argument('--window_size', type=int, default=4)
    args = parser.parse_args()

    # Define the names of the models you want to plot
    models = ['our model']  # Update this to match your model's name

    ax4 = None  # For reward plot

    for i, log_file in enumerate(args.log_files):
        with open(log_file, 'r') as file:
            log = file.read()

        # Simplified regex pattern
        train_pattern = r"TRAIN.*?in\s+episode\s+(\d+).*?total reward:\s+([-+]?\d+\.\d+)"
        train_episode = []
        train_reward = []
        for r in re.findall(train_pattern, log, re.DOTALL):
            train_episode.append(int(r[0]))
            train_reward.append(float(r[1]))

        # Debug: Print extracted episodes and rewards
        print(f"Extracted {len(train_episode)} training episodes from {log_file}")
        print(f"Training episodes: {train_episode[:10]}...")
        print(f"Training rewards: {train_reward[:10]}...")

        if len(train_episode) == 0:
            print(f"No training data found in log file {log_file}")
            continue

        # Adjust window size
        window_size = min(args.window_size, len(train_reward))

        # Smooth training data
        train_reward_smooth = running_mean(train_reward, window_size)

        # Adjust x-values for smoothed data
        x_values = train_episode[window_size - 1:]

        # Plot reward
        if args.plot_reward:
            if ax4 is None:
                fig4, ax4 = plt.subplots()
            if args.plot_train:
                # Plot unsmoothed data
                ax4.plot(train_episode, train_reward, label=f"{models[i]} (Raw)", alpha=0.3)
                # Plot smoothed data
                ax4.plot(x_values, train_reward_smooth, label=f"{models[i]} (Smoothed)")
            ax4.set_xlabel('Episodes')
            ax4.set_ylabel('Reward')
            ax4.set_title('Single Step Reward')
            ax4.set_xlim(1, max(train_episode))
            ax4.legend()

    # Save and show the reward plot
    if args.plot_reward and ax4 is not None:
        plt.savefig("./training_curve_2robots.png",dpi=300)
        plt.show()


if __name__ == '__main__':
    main()
