import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='FetchPushDense-v4', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=3, type=int, help='Random seed')
    parser.add_argument('--n-steps-per-rollout', default=4096, type=int, help='Number of steps per rollout')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
    parser.add_argument('--gamma', default=0.98, type=float, help='Discount factor')
    parser.add_argument('--ent-coef', default=0.0, type=float, help='Entropy coefficient')
    parser.add_argument('--learning-rate', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--clip-range', default=0.2, type=float, help='Clip range for PPO')
    parser.add_argument('--max-grad-norm', default=0.5, type=float, help='Max gradient norm')
    parser.add_argument('--n-epochs', default=10, type=int, help='Number of epochs per update')
    parser.add_argument('--gae-lambda', default=0.95, type=float, help='GAE lambda')
    parser.add_argument('--vf-coef', default=0.5, type=float, help='Value function coefficient')

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='logs/FetchPushDense-v4/', type=str, help='Base directory for tensorboard logs')

    parser.add_argument('--init-model-path', default='full_exp_on_ppo/models/FetchPushDense-v4/ppo_fetch_push_dense_1M', type=str, help='Base directory for init model')

    args = parser.parse_args(rest_args)

    return args