# -*- coding: utf8 -*-
import os
import numpy as np
import tensorflow as tf
from model import Model

from tensorflow.examples.tutorials.mnist import input_data

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

mnist = input_data.read_data_sets('data/')

input_size = 28
num_class = 10
num_layers = 2
seq_len = 28
batch_size = 200
num_units = 512

learn_rate = 0.001
step = 50000

model = Model(input_size=input_size,
              num_class=num_class,
              batch_size=batch_size,
              num_units=num_units,
              num_layers=num_layers)

op = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(model.loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    seq_lens = np.ones(batch_size) * seq_len
    for i in range(step):
        inputs, labels = mnist.train.next_batch(batch_size)
        inputs = np.reshape(inputs, (batch_size, seq_len, input_size))
        
        feed = {
            model.inputs : inputs,
            model.target : labels,
            model.seq_len : seq_lens,
            model.keep_prob : 0.5
        }
        sess.run([op, model.loss, model.acc], feed_dict=feed)

        if (i+1) % 100 == 0 :
            feed[model.keep_prob] = 1.0
            loss, acc = sess.run([model.loss, model.acc], feed_dict=feed)    
            msg = 'train step: %d, accuracy: %.4f, loss: %f' % (i, acc, loss)
            print(msg)

    inputs = np.reshape(mnist.test.images, (-1, seq_len, input_size))
    seq_lens = np.ones(inputs.shape[0]) * seq_len
    feed = {
        model.inputs : inputs,
        model.target : mnist.test.labels,
        model.seq_len : seq_lens,
        model.keep_prob : 1.0
    }
    loss, acc = sess.run([model.loss, model.acc], feed_dict=feed)    
    msg = 'test accuracy: %.4f, loss: %f' % (acc, loss)
    print(msg)