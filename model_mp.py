"""This is the Model File of MatchPyramid.

This module is used to construct the MatchPyramid described in paper https://arxiv.org/abs/1602.06359.
"""

__version__ = '0.1'
__author__ = 'Liang Pang'

import sys

import tensorflow as tf
import numpy as np

"""
Model Class
"""
class Model():

    def __init__(self, config):
        self.config = config
        tf.reset_default_graph()
        self.X1 = tf.placeholder(tf.int32, name='X1', shape=(None, config['data1_maxlen']))
        self.X2 = tf.placeholder(tf.int32, name='X2', shape=(None, config['data2_maxlen']))
        self.X1_len = tf.placeholder(tf.int32, name='X1_len', shape=(None, ))
        self.X2_len = tf.placeholder(tf.int32, name='X2_len', shape=(None, ))
        self.Y = tf.placeholder(tf.int32, name='Y', shape=(None, ))
        self.F = tf.placeholder(tf.float32, name='F', shape=(None, config['feat_size']))

        self.dpool_index = tf.placeholder(tf.int32, name='dpool_index', shape=(None, config['data1_maxlen'], config['data2_maxlen'], 3))

        self.batch_size = tf.shape(self.X1)[0]
        
        #self.embedding = tf.get_variable('embedding', initializer = config['embedding'], dtype=tf.float32, trainable=False)
        self.embedding = tf.get_variable('embedding', shape=(100,100), dtype=tf.float32, trainable=False)

        self.embed1 = tf.nn.embedding_lookup(self.embedding, self.X1)
        self.embed2 = tf.nn.embedding_lookup(self.embedding, self.X2)
        
        # batch_size * X1_maxlen * X2_maxlen
        self.cross = tf.einsum('abd,acd->abc', self.embed1, self.embed2)
        self.cross_img = tf.expand_dims(self.cross, 3)
        
        # convolution
        self.w1 = tf.get_variable('w1', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2, dtype=tf.float32) , dtype=tf.float32, shape=[2, 10, 1, 8])
        self.b1 = tf.get_variable('b1', initializer=tf.constant_initializer() , dtype=tf.float32, shape=[8])
        # batch_size * X1_maxlen * X2_maxlen * feat_out
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.cross_img, self.w1, [1, 1, 1, 1], "SAME") + self.b1)

        # dynamic pooling
        self.conv1_expand = tf.gather_nd(self.conv1, self.dpool_index)

        self.pool1 = tf.nn.max_pool(self.conv1_expand,
                        [1, config['data1_maxlen'] / config['data1_psize'], config['data2_maxlen'] / config['data2_psize'], 1], 
                        [1, config['data1_maxlen'] / config['data1_psize'], config['data2_maxlen'] / config['data2_psize'], 1], "VALID")

        with tf.variable_scope('fc1'):
            self.fc1 = tf.nn.relu(tf.contrib.layers.linear(tf.reshape(self.pool1, [self.batch_size, config['data1_psize'] * config['data2_psize'] * 8]), 20))

        self.pred = tf.contrib.layers.linear(self.fc1, 1)
        pos = tf.strided_slice(self.pred, [0], [self.batch_size], [2])
        neg = tf.strided_slice(self.pred, [1], [self.batch_size], [2])
        self.loss = tf.reduce_mean(tf.maximum(1.0 + neg - pos, 0.0))
    
        self.train_model = tf.train.AdamOptimizer().minimize(self.loss)
    
        self.saver = tf.train.Saver(max_to_keep=20)

    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i/stride1) for i in range(max_len1)]
            idx2_one = [int(i/stride2) for i in range(max_len2)]
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2,1,0))
            return index_one
        index = []
        for i in range(len(len1)):
            index.append(dpool_index_(i, len1[i], len2[i], max_len1, max_len2))
        return np.array(index)

    def init_step(self, sess):
        sess.run(tf.global_variables_initializer())

    def train_step(self, sess, feed_dict):
        feed_dict[self.dpool_index] = self.dynamic_pooling_index(feed_dict[self.X1_len], feed_dict[self.X2_len], 
                                            self.config['data1_maxlen'], self.config['data2_maxlen'])
        _, loss = sess.run([self.train_model, self.loss], feed_dict=feed_dict) 
        return loss

    def test_step(self, sess, feed_dict):
        feed_dict[self.dpool_index] = self.dynamic_pooling_index(feed_dict[self.X1_len], feed_dict[self.X2_len], 
                                            self.config['data1_maxlen'], self.config['data2_maxlen'])
        pred = sess.run(self.pred, feed_dict=feed_dict)
        return pred
    
    def eval_step(self, sess, node, feed_dict):
        feed_dict[self.dpool_index] = self.dynamic_pooling_index(feed_dict[self.X1_len], feed_dict[self.X2_len], 
                                            self.config['data1_maxlen'], self.config['data2_maxlen'])
        node_value = sess.run(node, feed_dict=feed_dict)
        return node_value
