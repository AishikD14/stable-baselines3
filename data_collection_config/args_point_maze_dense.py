import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env-name',
        default='PointMaze_UMazeDense-v3',
        type=str,
        help='Gym environment ID'
    )

    parser.add_argument(
        '--policy',
        default='MlpPolicy',
        type=str,
        help='Policy architecture'
    )

    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # PPO rollout
    parser.add_argument(
        '--n-steps-per-rollout',
        default=2048,
        type=int
    )

    parser.add_argument(
        '--batch-size',
        default=64,
        type=int
    )

    # Return / advantage estimation
    parser.add_argument(
        '--gamma',
        default=0.99,
        type=float
    )

    parser.add_argument(
        '--gae-lambda',
        default=0.95,
        type=float
    )

    # PPO optimization
    parser.add_argument(
        '--learning-rate',
        default=2e-4,
        type=float
    )

    parser.add_argument(
        '--clip-range',
        default=0.2,
        type=float
    )

    parser.add_argument(
        '--n-epochs',
        default=10,
        type=int
    )

    parser.add_argument(
        '--max-grad-norm',
        default=0.5,
        type=float
    )

    parser.add_argument(
        '--vf-coef',
        default=0.5,
        type=float
    )

    # Important for maze exploration
    parser.add_argument(
        '--ent-coef',
        default=0.01,
        type=float
    )

    parser.add_argument(
        '--device',
        default='cpu',
        type=str
    )

    parser.add_argument(
        '--tensorboard-log',
        default='logs/PointMaze_UMazeDense-v3/',
        type=str
    )

    parser.add_argument(
        '--init-model-path',
        default='full_exp_on_ppo/models/PointMaze_UMazeDense-v3/ppo_pointmaze_1M',
        type=str
    )

    args = parser.parse_args(rest_args)

    return args