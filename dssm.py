import tensorflow as tf
import numpy as np
from lstm import LSTM
import random


class Dssm(object):

    def __init__(self, vocab_size, 
        num_lstm_units,
        layer_num,
        batch_size, 
        negtive_size, 
        SOFTMAX_R, 
        l2_reg_lambda, 
        learning_rate, 
        is_cosin,
        embedding_size,
        max_document_length,
        initial_embeddings = None):

        global_step = tf.Variable(0, trainable=False)

        self.keep_prob = tf.placeholder(tf.float32)

        # random embeddings if pretrained embedding is None
        if initial_embeddings is None:
            initial_embeddings = tf.truncated_normal([vocab_size, embedding_size], 0.0, 1.0)
        self.lstm = LSTM(
            max_document_length,
            num_lstm_units,
            layer_num,
            batch_size,
            self.keep_prob,
            initial_embeddings)

        self.q_y = tf.nn.relu(self.lstm.output_x)
        self.qs_y = tf.nn.relu(self.lstm.output_y)

        with tf.name_scope('rotate'):

            temp = tf.tile(self.qs_y, [1, 1])
            for i in range(negtive_size):
                rand = int((random.random() + i) * batch_size / negtive_size)
                if rand == 0:
                    rand = rand + 1
                rand_qs_y1 = tf.slice(temp, [rand, 0], [batch_size - rand, -1])
                rand_qs_y2 = tf.slice(temp, [0, 0], [rand, -1])

                self.qs_y = tf.concat(axis = 0, values = [self.qs_y, rand_qs_y1, rand_qs_y2])

        with tf.name_scope('sim'):
            if is_cosin:
                # cosine similarity
                q_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.q_y), 1, True)), [negtive_size + 1, 1])
                qs_norm = tf.sqrt(tf.reduce_sum(tf.square(self.qs_y), 1, True))

                prod = tf.reduce_sum(tf.multiply(tf.tile(self.q_y, [negtive_size + 1, 1]), self.qs_y), 1, True)
                norm_prod = tf.multiply(q_norm, qs_norm)

                sim_raw = tf.truediv(prod, norm_prod)
            else:
                q_exp = tf.tile(self.q_y, [negtive_size + 1, 1])
                sim_raw = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(q_exp, self.qs_y)), 1, True))

            self.cos_sim = tf.transpose(tf.reshape(tf.transpose(sim_raw), [negtive_size + 1, batch_size])) * SOFTMAX_R

        with tf.name_scope('loss'):
            # train Loss
            self.prob = tf.nn.softmax(self.cos_sim)
            self.hit_prob = tf.slice(self.prob, [0, 0], [-1, 1])
            if is_cosin:
                raw_loss = -tf.reduce_sum(tf.log(self.hit_prob)) / batch_size
            else:
                raw_loss = tf.reduce_sum(tf.log(self.hit_prob)) / batch_size
            
            self.loss = raw_loss
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('training'):
            # optimizer
            self.lr = tf.train.exponential_decay(learning_rate, global_step, 200, 0.96, staircase=True)
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=global_step)# 

        with tf.name_scope('accuracy'):
            correct_prediction = tf.greater(self.hit_prob, 0.9)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)