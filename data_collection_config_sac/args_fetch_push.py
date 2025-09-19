import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='FetchPush-v4', type=str, help='Gym environment ID')
    parser.add_argument('--policy', default='MultiInputPolicy', type=str, help='Policy architecture')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--learning-starts', default=1000, type=int)

    parser.add_argument('--buffer-size', default=1000000, type=int)
    parser.add_argument('--ent-coef', default='auto', type=str)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--normalize', default=True, type=bool)

    parser.add_argument('--replay-buffer-class', default="HerReplayBuffer", type=str)
    parser.add_argument('--use-replay-buffer-kwargs', default=True, type=bool)
    parser.add_argument('--goal-selection-strategy', default="future", type=str)
    parser.add_argument('--n-sampled-goal', default=4, type=int)

    parser.add_argument('--use-policy-kwargs', default=True, type=bool)
    parser.add_argument('--net-arch', nargs='+', type=int, default=[64, 64])

    parser.add_argument('--device', default='cpu', type=str, help='Device to use: cpu or cuda')
    parser.add_argument('--tensorboard-log', default='sac_logs/FetchPush-v4/', type=str, help='Tensorboard log directory base')

    parser.add_argument('--init-model-path', default='full_exp_on_sac/models/FetchPush-v4/sac_fetch_push_19k', type=str, help='Base directory for init model')

    args = parser.parse_args(rest_args)

    return args
