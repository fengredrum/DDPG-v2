import tensorflow as tf

hid1_size = 200
hid2_size = 100


def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter


def build_actor(s, act_dim, is_training, reuse=None, getter=None):
    with tf.variable_scope('Actor', reuse=reuse, custom_getter=getter):
        init_w = tf.random_normal_initializer(0., 0.3)
        init_b = tf.constant_initializer(0.1)
        hidden_1 = tf.layers.dense(s, hid1_size, activation=tf.nn.relu,
                                   kernel_initializer=init_w, bias_initializer=init_b, name='hidden_1')
        hid1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor/hidden_1')
        h1w_summary = tf.summary.histogram('h1w', hid1_vars[0])
        h1b_summary = tf.summary.histogram('h1b', hid1_vars[1])
        h1out_summary = tf.summary.histogram('h1out', hidden_1)
        h1_bn = tf.layers.batch_normalization(hidden_1, training=is_training, name='bn_1')
        h1bn_summary = tf.summary.histogram('h1bn', h1_bn)
        hidden_2 = tf.layers.dense(h1_bn, hid2_size, activation=tf.nn.relu,
                                   kernel_initializer=init_w, bias_initializer=init_b, name='hidden_2')
        hid2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor/hidden_2')
        h2w_summary = tf.summary.histogram('h2w', hid2_vars[0])
        h2b_summary = tf.summary.histogram('h2b', hid2_vars[1])
        h2out_summary = tf.summary.histogram('h2out', hidden_2)
        h2_bn = tf.layers.batch_normalization(hidden_2, training=is_training, name='bn_2')
        h2bn_summary = tf.summary.histogram('h2bn', h2_bn)
        actions = tf.layers.dense(h2_bn, act_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                  bias_initializer=init_b, name='action')
        action_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor/action')
        actw_summary = tf.summary.histogram('actw', action_vars[0])
        actb_summary = tf.summary.histogram('actb', action_vars[1])
        actout_summary = tf.summary.histogram('actout', actions)
    return actions, tf.summary.merge([h1w_summary, h1b_summary, h1out_summary, h1bn_summary,
                                      h2w_summary, h2b_summary, h2out_summary, h2bn_summary,
                                      actw_summary, actb_summary, actout_summary])


def build_critic(s, a, act_dim, reuse=None, getter=None):
    with tf.variable_scope('Critic', reuse=reuse, custom_getter=getter):
        init_w = tf.random_normal_initializer(0., 0.1)
        init_b = tf.constant_initializer(0.1)
        hidden_1 = tf.layers.dense(s, hid1_size, activation=tf.nn.relu,
                                   kernel_initializer=init_w, bias_initializer=init_b, name='hidden_1')
        hid1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic/hidden_1')
        h1w_summary = tf.summary.histogram('h1w', hid1_vars[0])
        h1b_summary = tf.summary.histogram('h1b', hid1_vars[1])
        h1out_summary = tf.summary.histogram('h1out', hidden_1)
        with tf.variable_scope('hidden_2'):
            w2_s = tf.get_variable('w2_s', [hid1_size, hid2_size], initializer=init_w)
            h2ws_summary = tf.summary.histogram('h2ws', w2_s)
            w2_a = tf.get_variable('w2_a', [act_dim, hid2_size], initializer=init_w)
            h2wa_summary = tf.summary.histogram('h2wa', w2_a)
            b2 = tf.get_variable('b2', [1, hid2_size], initializer=init_b)
            b2_summary = tf.summary.histogram('b2', b2)
            hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2_s) + tf.matmul(a, w2_a) + b2)
            h2out_summary = tf.summary.histogram('h2out', hidden_2)
        q = tf.layers.dense(hidden_2, 1, kernel_initializer=init_w, bias_initializer=init_b, name='q')
        q_summary = tf.summary.histogram('Q', q)
    return q, tf.summary.merge([h1w_summary, h1b_summary, h1out_summary,
                                h2ws_summary, h2wa_summary, b2_summary, h2out_summary,
                                q_summary])
