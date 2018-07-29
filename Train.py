import gym
from arguments import get_args
from ddpg import *

np.random.seed(1)
tf.set_random_seed(1)


def train():
    args = get_args()
    env = gym.make(args.env_name)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    agent = DDPG(sess, env, args)
    learn(args, env, agent)
    print('Environment: ', args.env_name)


if __name__ == '__main__':
    start = time.clock()
    train()
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
