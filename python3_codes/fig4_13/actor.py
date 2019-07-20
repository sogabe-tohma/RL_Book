"""
Dependencies:
tensorflow  1.2
"""
import numpy as np
import tensorflow as tf


class Actor():
    def __init__(self, sess, State_dim, Action_dim, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, State_dim], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope('Actor'):
            layer1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=layer1,
                units=Action_dim,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, s, a, td):

        feed_dict = {self.s: [s], self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        probs = self.sess.run(self.acts_prob, {self.s: [s]})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
