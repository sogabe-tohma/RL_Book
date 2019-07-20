"""
Dependencies:
tensorflow 1.2
keras 2.2.4
"""
import os
import shutil
import random
import numpy as np
import tensorflow as tf
from collections import deque

class Memory:
    def __init__(self, capacity, dims):
        self.cap=capacity
        self.capacity = np.zeros((capacity, dims))
        self.memo=deque(maxlen=48)
        self.memory_counter  = 0
        self.t_memory=0

    def store(self, s, a,r, s_):
        if self.memory_counter==self.cap:
            self.memory_counter = 0
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % self.cap
        self.capacity[index, :] = transition
        self.memory_counter += 1
        if self.t_memory<=self.cap-1:
            self.t_memory+=1
        else:
            self.t_memory=self.cap-1

    def sample(self, n):
        if len(self.capacity) > n:
            indices = np.random.choice(self.t_memory, size=n)
        return self.capacity[indices, :]

class Policy:
    def __init__(self, state_size, action_size):
        tf.reset_default_graph()
        self.n_features = state_size
        self.n_actions = action_size
        self.memory_size = 10000
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.lr = 0.0001
        self.batch=48
        self.output_graph=False
        self.learn_step_counter=0
        self.replace_target_iter=48
        self._build_net()
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        self.sess = tf.Session()
        if self.output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.Memory = Memory(self.memory_size, self.n_features * 2 + 2)
        self.saver=tf.train.Saver()

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')

        w_initializer, b_initializer = tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 30, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_el = tf.layers.dense(e1, 10, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='e2')
            self.q_eval=tf.layers.dense(self.q_el,self.n_actions,
                                        kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e3')

        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 30, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.t_el = tf.layers.dense(t1,10, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')
            self.q_next=tf.layers.dense(self.t_el,self.n_actions,
                                            kernel_initializer=w_initializer,bias_initializer=b_initializer, name='t3')
        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation):
        self.action=None
        if np.random.rand()>= self.epsilon :
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [observation]})
            action = np.argmax(actions_value)
        else:
            action=np.random.randint(4,size=1)[0]
        self.action=action
        return self.action

    def learn_act(self,observation,reward,next_state):
        self.Memory.store(observation,reward,self.action,next_state)
        if self.batch<self.Memory.t_memory:
            batch_memory=self.Memory.sample(self.batch)

            _, cost = self.sess.run(
                [self._train_op, self.loss],
                feed_dict={
                    self.s: batch_memory[:, 0:12],
                    self.a: batch_memory[:, 13],
                    self.r: batch_memory[:, 12],
                    self.s_: batch_memory[:, 14:26],
                })
            self.cost_his.append(cost)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

    def save_model(self):
        path="./model_save/"
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.sess, path+"/model.ckpt")

    def test_model(self,state):
        path="./model_save/model.ckpt"
        self.saver.restore(self.sess, path)
