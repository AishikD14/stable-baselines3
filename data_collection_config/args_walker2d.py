import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='Walker2d-v5', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--n-steps-per-rollout', default=512, type=int, help='Number of steps per rollout')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--gae-lambda', default=0.95, type=float, help='GAE lambda')
    parser.add_argument('--ent-coef', default=0.000585045, type=float, help='Entropy coefficient')
    parser.add_argument('--learning-rate', default=5.05041e-05, type=float, help='Learning rate')
    parser.add_argument('--clip-range', default=0.1, type=float, help='Clip range')
    parser.add_argument('--max-grad-norm', default=1, type=float, help='Max gradient norm')
    parser.add_argument('--n-epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument('--vf-coef', default=0.871923, type=float, help='Value function coefficient')

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='logs/Walker2d-v5/', type=str, help='Tensorboard log directory base')

    parser.add_argument('--init-model-path', default='full_exp_on_ppo/models/Walker2d-v5/ppo_walker2d_1M', type=str, help='Base directory for init model')

    args = parser.parse_args(rest_args)

    return args
