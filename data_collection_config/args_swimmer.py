import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='Swimmer-v5', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--n-envs', default=4, type=int, help='Number of envs')
    parser.add_argument('--n-steps-per-rollout', default=1024, type=int, help='Number of steps per rollout')
    parser.add_argument('--batch-size', default=256, type=int, help='Batch size')
    parser.add_argument('--gamma', default=0.9999, type=float, help='Discount factor')
    parser.add_argument('--learning-rate', default=6e-4, type=float, help='Learning rate')
    parser.add_argument('--n-epochs', default=10, type=int, help='Number of epochs per update')
    parser.add_argument('--gae-lambda', default=0.98, type=float, help='GAE lambda')

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='logs/Swimmer-v5/', type=str, help='Base directory for tensorboard logs')

    parser.add_argument('--init-model-path', default='full_exp_on_ppo/models/Swimmer-v5/ppo_swimmer_1M', type=str, help='Base directory for init model')

    args = parser.parse_args(rest_args)

    return args