# -*- coding: utf8 -*-
import tensorflow as tf

class Model(object):
    def __init__(self, input_size, num_class, num_layers, batch_size, num_units):
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_class = num_class
        self.input_size = input_size
        self.batch_size = batch_size
        self.build()

    def build(self):
        self.inputs = tf.placeholder(tf.float32, [None, None, self.input_size])
        self.target = tf.sparse_placeholder(tf.int32, name='label')
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)


        m_cell = tf.nn.rnn_cell.MultiRNNCell([self.unit() for _ in range(self.num_layers)])
        output, _ = tf.nn.dynamic_rnn(m_cell, self.inputs, self.seq_len, dtype=tf.float32, time_major=False)
        h_state = tf.reshape(output, (-1, self.num_units))

        w = tf.Variable(tf.truncated_normal([self.num_units, self.num_class], stddev=0.1))
        b = tf.constant(0.1, dtype=tf.float32, shape=[self.num_class])

        logits = tf.matmul(h_state, w) + b
        logits = tf.reshape(logits, [self.batch_size, -1, self.num_class])
        self.logits = tf.transpose(logits, (1, 0, 2))

        self.decoded, _ = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len, merge_repeated=False)

        self.cost = tf.nn.ctc_loss(labels=self.target, inputs=self.logits, sequence_length=self.seq_len)
        self.loss = tf.reduce_mean(self.cost)

        self.op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)        
        
        self.err = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.target))


    def unit(self):
        rnn_cell = tf.nn.rnn_cell.LSTMCell(self.num_units)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.keep_prob)
        return rnn_cell