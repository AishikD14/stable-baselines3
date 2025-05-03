import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='MountainCar-v0', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--n-envs', default=16, type=int, help='Number of parallel environments')
    parser.add_argument('--n-steps-per-rollout', default=16, type=int, help='Number of steps per rollout')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--ent-coef', default=0.0, type=float, help='Entropy coefficient')
    parser.add_argument('--n-epochs', default=4, type=int, help='Number of epochs per update')
    parser.add_argument('--gae-lambda', default=0.8, type=float, help='GAE lambda')
    parser.add_argument('--normalize', default=True, type=bool, help='Normalize observations and rewards')

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='logs/MountainCar-v0/', type=str, help='Base directory for tensorboard logs')

    parser.add_argument('--init-model-path', default='full_exp_on_ppo/models/MountainCar-v0/ppo_mountain_car_1M', type=str, help='Base directory for init model')

    args = parser.parse_args(rest_args)

    return args