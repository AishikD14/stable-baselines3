import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='Humanoid-v5', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--n-steps-per-rollout', default=512, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--gae-lambda', default=0.9, type=float)
    parser.add_argument('--ent-coef', default=0.00238306, type=float)
    parser.add_argument('--learning-rate', default=3.56987e-05, type=float)
    parser.add_argument('--clip-range', default=0.3, type=float)
    parser.add_argument('--max-grad-norm', default=2, type=float)
    parser.add_argument('--n-epochs', default=5, type=int)
    parser.add_argument('--vf-coef', default=0.431892, type=float)

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='logs/Humanoid-v5/', type=str, help='Tensorboard log directory base')

    parser.add_argument('--init-model-path', default='full_exp_on_ppo/models/Humanoid-v5/ppo_humanoid_1M', type=str, help='Base directory for init model')

    # Optional policy kwargs
    parser.add_argument('--use-policy-kwargs', default=True, type=bool, help='Enable custom policy_kwargs')
    parser.add_argument('--pi-layers', nargs='+', type=int, default=[256, 256], help='Hidden layers for policy')
    parser.add_argument('--vf-layers', nargs='+', type=int, default=[256, 256], help='Hidden layers for value function')
    parser.add_argument('--activation-fn', default='ReLU', choices=['ReLU', 'Tanh', 'LeakyReLU'])
    parser.add_argument('--log-std-init', default=-2, type=float, help='Initial log standard deviation')
    parser.add_argument('--ortho-init', default=False, type=bool, help='Disable orthogonal initialization (default: True)')

    args = parser.parse_args(rest_args)

    return args
