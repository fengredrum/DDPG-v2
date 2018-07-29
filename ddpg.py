import time
import numpy as np
from replayBuffer import Memory
from network import *


class DDPG:
    """docstring for DDPG"""

    def __init__(self, sess, env, args):
        self.s_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.sess = sess
        self.args = args

        self._build_graph()
        self.memory = Memory(self.args.replayBuffer_size, dims=2 * self.s_dim + self.act_dim + 1)

    def _build_graph(self):
        self._placehoders()
        self._actor_critic()
        self._loss_train_op()
        self.score = tf.Variable(0., trainable=False, dtype=tf.float32, name='score')
        self.score_summary = tf.summary.scalar('score', self.score)
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter('logs/')
        self.writer.add_graph(self.sess.graph)

    def _placehoders(self):
        with tf.name_scope('inputs'):
            self.current_state = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s')
            self.reward = tf.placeholder(tf.float32, [None, 1], name='r')
            self.next_state = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s_')
            self.is_training = tf.placeholder(tf.bool, name='is_training')

    def _actor_critic(self):
        self.actor, self.actor_summary = build_actor(self.current_state, self.act_dim, self.is_training)
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor')
        actor_ema = tf.train.ExponentialMovingAverage(decay=1 - self.args.tau)
        self.update_targetActor = actor_ema.apply(self.actor_vars)
        self.targetActor, _ = build_actor(self.next_state, self.act_dim, False,
                                          reuse=True, getter=get_getter(actor_ema))

        self.critic, self.critic_summary = build_critic(self.current_state, self.actor, self.act_dim)
        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic')
        critic_ema = tf.train.ExponentialMovingAverage(decay=1 - self.args.tau)
        self.update_targetCritic = critic_ema.apply(self.critic_vars)
        self.targetCritic, _ = build_critic(self.next_state, self.targetActor, self.act_dim,
                                            reuse=True, getter=get_getter(critic_ema))

    def _loss_train_op(self):
        max_grad = 2
        with tf.variable_scope('target_q'):
            self.target_q = self.reward + self.args.gamma * self.targetCritic
        with tf.variable_scope('TD_error'):
            self.critic_loss = tf.squared_difference(self.target_q, self.critic)
        with tf.variable_scope('critic_grads'):
            self.critic_grads = tf.gradients(ys=self.critic_loss, xs=self.critic_vars)
            for ix, grad in enumerate(self.critic_grads):
                self.critic_grads[ix] = grad / self.args.batch_size
        with tf.variable_scope('C_train'):
            critic_optimizer = tf.train.AdamOptimizer(self.args.critic_lr, epsilon=1e-5)
            self.train_critic = critic_optimizer.apply_gradients(zip(self.critic_grads, self.critic_vars))
        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.critic, self.actor)[0]
        with tf.variable_scope('actor_grads'):
            self.actor_grads = tf.gradients(ys=self.actor, xs=self.actor_vars, grad_ys=self.a_grads)
            for ix, grad in enumerate(self.actor_grads):
                self.actor_grads[ix] = tf.clip_by_norm(grad / self.args.batch_size, max_grad)
        with tf.variable_scope('A_train'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                actor_optimizer = tf.train.AdamOptimizer(-self.args.actor_lr,
                                                         epsilon=1e-5)  # (- learning rate) for ascent policy
                self.train_actor = actor_optimizer.apply_gradients(zip(self.actor_grads, self.actor_vars))

    def choose_action(self, state):
        state = state[np.newaxis, :]  # single state
        return self.sess.run(self.actor, feed_dict={self.current_state: state,
                                                    self.is_training: False})[0]  # single action

    def train(self, episode=None, ep_reward=None):
        b_m = self.memory.sample(self.args.batch_size)
        b_s = b_m[:, :self.s_dim]
        b_a = b_m[:, self.s_dim: self.s_dim + self.act_dim]
        b_r = b_m[:, -self.s_dim - 1: -self.s_dim]
        b_s_ = b_m[:, -self.s_dim:]

        if episode is None:
            critic_feed_dict = {self.current_state: b_s, self.actor: b_a, self.reward: b_r, self.next_state: b_s_}
            self.sess.run([self.train_critic, self.update_targetCritic],
                          feed_dict=critic_feed_dict)
            actor_feed_dict = {self.current_state: b_s, self.next_state: b_s_, self.is_training: True}
            self.sess.run([self.train_actor, self.update_targetActor],
                          feed_dict=actor_feed_dict)
        else:
            update_score = self.score.assign(tf.convert_to_tensor(ep_reward, dtype=tf.float32))
            with tf.control_dependencies([update_score]):
                merged_score = tf.summary.merge([self.score_summary])
            critic_feed_dict = {self.current_state: b_s, self.actor: b_a, self.reward: b_r, self.next_state: b_s_}
            _, _, critic = self.sess.run([self.train_critic, self.update_targetCritic, self.critic_summary],
                                         feed_dict=critic_feed_dict)
            self.writer.add_summary(critic, episode)
            actor_feed_dict = {self.current_state: b_s, self.next_state: b_s_, self.is_training: True}
            merged = tf.summary.merge([merged_score, self.actor_summary])
            _, _, actor = self.sess.run([self.train_actor, self.update_targetActor, merged],
                                        feed_dict=actor_feed_dict)
            self.writer.add_summary(actor, episode)

    def perceive(self, state, action, reward, next_state, episode=None, ep_reward=None):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.memory.store_transition(state, action, reward, next_state)
        # Store transitions to replay start size then start training
        if self.memory.pointer > self.args.replayBuffer_size:
            self.train(episode, ep_reward)


def learn(args, env, agent):
    render = False
    var = 3  # control exploration
    start = time.time()
    for e in range(args.num_episodes):
        obs = env.reset()
        ep_reward = 0
        if var >= 0.1:
            var *= .995  # decay the action randomness
        for j in range(args.num_steps):
            if render:
                env.render()
            action = agent.choose_action(obs)
            # Add exploration noise
            action = np.clip(np.random.normal(action, var), -2, 2)  # add randomness to action selection for exploration
            next_obs, reward, done, info = env.step(action)
            ep_reward += reward
            if j == args.num_steps - 1:
                agent.perceive(obs, action, reward * 0.1, next_obs, episode=e, ep_reward=ep_reward)
                end = time.time()
                total_num_steps = (e + 1) * args.num_steps
                print('Episode:', e, 'FPS:', int(total_num_steps / (end - start)),
                      'Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > 10000:
                    render = True
                break
            else:
                agent.perceive(obs, action, reward * 0.1, next_obs)
            obs = next_obs

    agent.sess.close()
