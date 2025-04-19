import argparse

parser = argparse.ArgumentParser('Parse configuration file')
parser.add_argument('--env_config', type=str, default='configs/env.config')
parser.add_argument('--policy', type=str, default='cadrl')
parser.add_argument('--policy_config', type=str, default='configs/policy.config')
parser.add_argument('--train_config', type=str, default='configs/train.config')
parser.add_argument('--output_dir', type=str, default='data43_hsarl/output')
parser.add_argument('--weights', type=str)
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--gpu', default=False, action='store_true')
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--visualize', default=False, action='store_true')
parser.add_argument('--phase', type=str, default='test')
parser.add_argument('--test_case', type=int, default=None)
parser.add_argument('--square', default=False, action='store_true')
parser.add_argument('--circle', default=False, action='store_true')
parser.add_argument('--video_file', type=str, default=None)
parser.add_argument('--traj', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--hose', default=False, action='store_true') # 有无软管,如果未指定该参数，默认状态下其值为False；若指定该参数，将该参数置为 True

args = parser.parse_args()