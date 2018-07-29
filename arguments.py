import argparse


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--env-name', default='HalfCheetah-v2',
                        help='environment to train on (default: HalfCheetah-v2)')
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='number of environment steps (default: 1000)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-episodes', type=int, default=int(5e0),
                        help='number of frames to train (default: 1e3)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.01,
                        help='discount factor for rewards (default: 0.01)')
    parser.add_argument('--critic-lr', type=float, default=5e-4,
                        help='critic learning rate (default: 2e-4)')
    parser.add_argument('--actor-lr', type=float, default=1e-4,
                        help='actor learning rate (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='number of batch size to train (default: 64)')
    parser.add_argument('--replayBuffer-size', type=int, default=int(2e2),
                        help='size of the replay buffer (default: 2e4)')
    parser.add_argument('--log-dir', default='logs/',
                        help='directory to save agent logs (default: "logs/")')
    args = parser.parse_args()

    return args
