# -*- coding: utf8 -*-
import tensorflow as tf

class Model(object):
    """docstring for Model"""
    def __init__(self, input_size, num_class, num_layers, batch_size, num_units):
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_class = num_class
        self.input_size = input_size
        self.batch_size = batch_size
        self.build()

    def build(self):
        self.inputs = tf.placeholder(tf.float32, [None, None, self.input_size])
        self.target = tf.placeholder(tf.int64, [None])
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        m_cell = tf.nn.rnn_cell.MultiRNNCell([self.unit() for _ in range(self.num_layers)])
        # init_state = m_cell.zero_state(self.batch_size, dtype=tf.float32)
        output, _ = tf.nn.dynamic_rnn(m_cell, self.inputs, self.seq_len, dtype=tf.float32)
        h_state = output[:, -1, :]

        w = tf.Variable(tf.truncated_normal([self.num_units, self.num_class], stddev=0.1))
        b = tf.constant(0.1, dtype=tf.float32, shape=[self.num_class])

        logits = tf.matmul(h_state, w) + b
        # print(tf.shape(logits))
        # print(tf.shape(self.target))

        # logits = tf.nn.relu(tf.matmul(h_state, w) + b)
        # logits = tf.nn.softmax(logits)
        self.loss = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.target))
        
        self.pre = tf.argmax(logits, 1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pre, self.target), tf.float32))


    def unit(self):
        rnn_cell =  tf.nn.rnn_cell.GRUCell(self.num_units)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.keep_prob)
        return rnn_cell