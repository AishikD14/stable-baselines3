import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='Pendulum-v1', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--n-envs', default=2, type=int, help='Number of parallel environments')
    parser.add_argument('--n-steps-per-rollout', default=1024, type=int, help='Number of steps per rollout')
    parser.add_argument('--gamma', default=0.9, type=float, help='Discount factor')
    parser.add_argument('--use-sde', default=True, type=bool, help='Use State Dependent Exploration (SDE)')
    parser.add_argument('--sde-sample-freq', default=4, type=int, help='SDE sample frequency')
    parser.add_argument('--n-critic-updates', default=15, type=int, help='Number of critic updates per iteration')

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='trpo_logs/Pendulum-v1/', type=str, help='Base directory for tensorboard logs')

    parser.add_argument('--init-model-path', default='full_exp_on_trpo/models/Pendulum-v1/trpo_pendulum_40k', type=str, help='Base directory for init model')

    args = parser.parse_args(rest_args)

    return args