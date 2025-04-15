import re
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

def running_mean(x, n):
    """Calculate running mean with window size n."""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def process_rewards(rewards, window_size=100, smooth_factor=5):
    """Process rewards with enhanced trend visualization."""
    # Calculate initial running mean
    smoothed = running_mean(rewards, window_size)
    
    # Apply additional Gaussian smoothing to highlight trend
    smoothed = gaussian_filter1d(smoothed, smooth_factor)
    
    # Pad the beginning to match original length
    pad_length = len(rewards) - len(smoothed)
    if pad_length > 0:
        smoothed = np.pad(smoothed, (pad_length, 0), mode='edge')
    
    return smoothed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+')
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--smooth_factor', type=int, default=5)
    args = parser.parse_args()

    # Set up plot style
    plt.style.use('seaborn')
    matplotlib.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'lines.linewidth': 1.5,
        'grid.alpha': 0.3
    })

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, log_file in enumerate(args.log_files):
        with open(log_file, 'r') as file:
            log = file.read()

        train_pattern = r"TRAIN in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
                       r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
                       r"total reward: (?P<reward>[-+]?\d+.\d+)"
        
        episodes = []
        rewards = []
        for r in re.findall(train_pattern, log):
            episodes.append(int(r[0]))
            rewards.append(float(r[4]))

        episodes = np.array(episodes)
        rewards = np.array(rewards)

        # Process rewards with enhanced trend visualization
        smoothed_rewards = process_rewards(rewards, 
                                         args.window_size, 
                                         args.smooth_factor)

        # Calculate confidence intervals with reduced noise
        window_size = args.window_size
        std_dev = np.array([np.std(rewards[max(0, i-window_size):i+1]) * 0.7  # Reduced multiplier
                           for i in range(len(rewards))])
        std_dev = gaussian_filter1d(std_dev, args.smooth_factor)  # Smooth the confidence bands
        
        upper_bound = smoothed_rewards + std_dev
        lower_bound = smoothed_rewards - std_dev

        # Plot with enhanced visibility
        ax.plot(episodes, smoothed_rewards, color='#1f77b4', 
                label='our model', zorder=3)
        ax.fill_between(episodes, lower_bound, upper_bound, 
                       color='#1f77b4', alpha=0.15)  # Reduced alpha for clearer trend

    # Customize plot appearance for better trend visibility
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.set_title('Cumulative Discounted Reward')
    
    ax.legend(loc='upper left')
    
    # Adjust y-axis limits to focus on the trend
    ymin, ymax = ax.get_ylim()
    y_padding = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - y_padding, ymax + y_padding)
    
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()