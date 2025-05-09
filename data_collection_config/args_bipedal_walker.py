import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='BipedalWalker-v3', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--n-envs', default=32, type=int, help='Number of parallel environments')
    parser.add_argument('--n-steps-per-rollout', default=2048, type=int, help='Number of steps per rollout')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('--gamma', default=0.999, type=float, help='Discount factor')
    parser.add_argument('--ent-coef', default=0.0, type=float, help='Entropy coefficient')
    parser.add_argument('--learning-rate', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--clip-range', default=0.18, type=float, help='Clip range for PPO')
    parser.add_argument('--n-epochs', default=10, type=int, help='Number of epochs per update')
    parser.add_argument('--gae-lambda', default=0.95, type=float, help='GAE lambda')
    parser.add_argument('--normalize', default=True, type=bool, help='Normalize observations')

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='logs/BipedalWalker-v3/', type=str, help='Base directory for tensorboard logs')

    parser.add_argument('--init-model-path', default='full_exp_on_ppo/models/BipedalWalker-v3/ppo_bipedal_walker_1M', type=str, help='Base directory for init model')

    args = parser.parse_args(rest_args)

    return args