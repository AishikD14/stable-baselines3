import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default='Walker2d-v5', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--n-steps-per-rollout', default=2048, type=int, help='Number of steps per rollout')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--gae-lambda', default=0.95, type=float, help='GAE lambda')
    parser.add_argument('--ent-coef', default=1e-4, type=float, help='Entropy coefficient')
    parser.add_argument('--learning-rate', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--clip-range', default=0.2, type=float, help='Clip range')
    parser.add_argument('--max-grad-norm', default=0.5, type=float, help='Max gradient norm')
    parser.add_argument('--n-epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--vf-coef', default=0.5, type=float, help='Value function coefficient')

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='logs/Walker2d-v5/', type=str, help='Tensorboard log directory base')

    parser.add_argument('--init-model-path', default='full_exp_on_ppo/models/Walker2d-v5/ppo_walker2d_1M', type=str, help='Base directory for init model')

    # Optional policy kwargs
    parser.add_argument('--use-policy-kwargs', action='store_true', help='Enable custom policy_kwargs')
    parser.add_argument('--pi-layers', nargs='+', type=int, default=[256, 256], help='Hidden layers for policy')
    parser.add_argument('--vf-layers', nargs='+', type=int, default=[256, 256], help='Hidden layers for value function')
    parser.add_argument('--activation-fn', default='ReLU', type=str, choices=['ReLU', 'Tanh', 'LeakyReLU'], help='Activation function for policy network')

    args = parser.parse_args(rest_args)

    return args
