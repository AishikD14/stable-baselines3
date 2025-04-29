import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='Humanoid-v5', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--n-steps-per-rollout', default=4096, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--gamma', default=0.995, type=float)
    parser.add_argument('--gae-lambda', default=0.92, type=float)
    parser.add_argument('--ent-coef', default=0.001, type=float)
    parser.add_argument('--learning-rate', default=2.5e-4, type=float)
    parser.add_argument('--clip-range', default=0.15, type=float)
    parser.add_argument('--max-grad-norm', default=0.3, type=float)
    parser.add_argument('--n-epochs', default=5, type=int)
    parser.add_argument('--vf-coef', default=0.7, type=float)

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='logs/Humanoid-v5/', type=str, help='Tensorboard log directory base')

    parser.add_argument('--init-model-path', default='full_exp_on_ppo/models/Humanoid-v5/ppo_humanoid_1M', type=str, help='Base directory for init model')

    # Optional policy kwargs
    parser.add_argument('--use-policy-kwargs', action='store_true', help='Enable custom policy_kwargs')
    parser.add_argument('--pi-layers', nargs='+', type=int, default=[512, 512], help='Hidden layers for policy')
    parser.add_argument('--vf-layers', nargs='+', type=int, default=[512, 512], help='Hidden layers for value function')
    parser.add_argument('--activation-fn', default='ReLU', choices=['ReLU', 'Tanh', 'LeakyReLU'])
    parser.add_argument('--log-std-init', default=-0.5, type=float, help='Initial log standard deviation')
    parser.add_argument('--ortho-init', action='store_false', help='Disable orthogonal initialization (default: True)')

    args = parser.parse_args(rest_args)

    return args
