import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='Humanoid-v5', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--n-steps-per-rollout', default=512, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--learning-starts', default=10000, type=int)
    # parser.add_argument('--n-steps-per-rollout', default=10512, type=int, help='Number of steps per rollout')  #For Baseline calculation search=1
    # parser.add_argument('--batch-size', default=657, type=int, help='Batch size') # For Baseline calculation search=1
    # parser.add_argument('--n-steps-per-rollout', default=5512, type=int, help='Number of steps per rollout')  #For Baseline calculation search=2
    # parser.add_argument('--batch-size', default=345, type=int, help='Batch size') # For Baseline calculation search=2
    # parser.add_argument('--n-steps-per-rollout', default=3845, type=int, help='Number of steps per rollout')  #For Baseline calculation search=3
    # parser.add_argument('--batch-size', default=240, type=int, help='Batch size') # For Baseline calculation search=3
    # parser.add_argument('--gamma', default=0.98, type=float)
    # parser.add_argument('--gae-lambda', default=0.8, type=float)
    # parser.add_argument('--ent-coef', default=4.9646e-07, type=float)
    # parser.add_argument('--learning-rate', default=1.90609e-05, type=float)
    # parser.add_argument('--clip-range', default=0.1, type=float)
    # parser.add_argument('--max-grad-norm', default=0.6, type=float)
    # parser.add_argument('--n-epochs', default=10, type=int)
    # parser.add_argument('--vf-coef', default=0.677239, type=float)

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='sac_logs/Humanoid-v5/', type=str, help='Tensorboard log directory base')

    parser.add_argument('--init-model-path', default='full_exp_on_sac/models/Humanoid-v5/sac_humanoid_1M', type=str, help='Base directory for init model')

    args = parser.parse_args(rest_args)

    return args
