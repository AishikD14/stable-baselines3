import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env-name', default='Hopper-v5', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--n-steps-per-rollout', default=512, type=int, help='Number of steps per rollout')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    # parser.add_argument('--n-steps-per-rollout', default=5512, type=int, help='Number of steps per rollout')  #For Baseline calculation search=2
    # parser.add_argument('--batch-size', default=345, type=int, help='Batch size') # For Baseline calculation search=2
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--gae-lambda', default=0.99, type=float, help='GAE lambda')
    parser.add_argument('--ent-coef', default=0.002295, type=float, help='Entropy coefficient')
    parser.add_argument('--learning-rate', default=9.808e-05, type=float, help='Learning rate')
    parser.add_argument('--clip-range', default=0.2, type=float, help='Clip range for PPO')
    parser.add_argument('--max-grad-norm', default=0.7, type=float, help='Max gradient norm')
    parser.add_argument('--n-epochs', default=5, type=int, help='Number of epochs per update')
    parser.add_argument('--vf-coef', default=0.8357, type=float, help='Value function coefficient')

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='logs/Hopper-v5/', type=str, help='Base directory for tensorboard logs')

    parser.add_argument('--init-model-path', default='full_exp_on_ppo/models/Hopper-v5/ppo_hopper_1M', type=str, help='Base directory for init model')

    # Policy kwargs
    parser.add_argument('--use-policy-kwargs', default=True, type=bool, help='Enable custom policy_kwargs')
    parser.add_argument('--pi-layers', nargs='+', type=int, default=[256, 256], help='Hidden layer sizes for policy network')
    parser.add_argument('--vf-layers', nargs='+', type=int, default=[256, 256], help='Hidden layer sizes for value network')
    parser.add_argument('--activation-fn', default='ReLU', type=str, choices=['ReLU', 'Tanh', 'LeakyReLU'], help='Activation function name')
    parser.add_argument('--log-std-init', default=-2, type=float, help='Initial log standard deviation')
    parser.add_argument('--ortho-init', default=False, type=bool, help='Disable orthogonal initialization (default: True)')

    # Normalize kwargs
    parser.add_argument('--use-normalize-kwargs', default=True, type=bool, help='Enable custom normalize_kwargs')
    parser.add_argument('--norm-obs', default=True, type=bool, help='Normalize observations')
    parser.add_argument('--norm-reward', default=False, type=bool, help='Normalize rewards')

    args = parser.parse_args(rest_args)

    return args
