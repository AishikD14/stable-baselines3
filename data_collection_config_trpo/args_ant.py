import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='Ant-v5', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--n-envs', default=2, type=int, help='Number of envs')
    parser.add_argument('--n-steps-per-rollout', default=1024, type=int, help='Number of steps per rollout')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--learning-rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--gae-lambda', default=0.95, type=float, help='GAE lambda')
    parser.add_argument('--normalize', default=True, type=bool, help='Normalize observations')
    parser.add_argument('--sub-sampling-factor', default=1, type=int, help='Subsampling factor for data collection')
    parser.add_argument('--cg-max-steps', default=25, type=int, help='Max steps for conjugate gradient')
    parser.add_argument('--cg-damping', default=0.1, type=float, help='Damping for conjugate gradient')
    parser.add_argument('--n-critic-updates', default=20, type=int, help='Number of critic updates per iteration')

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='trpo_logs/Ant-v5/', type=str, help='Base directory for tensorboard logs')

    parser.add_argument('--init-model-path', default='full_exp_on_trpo/models/Ant-v5/trpo_ant_1M', type=str, help='Base directory for init model')

    args = parser.parse_args(rest_args)

    return args