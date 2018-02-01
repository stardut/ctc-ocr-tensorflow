# -*- coding: utf8 -*-
import os
import numpy as np
import tensorflow as tf
from model import Model

from data import Data

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

input_size = 28
num_class = 10 + 1
num_layers = 2
seq_len = 28
batch_size = 64
num_units = 512
word_size = 4

learn_rate = 0.001
step = 50000

model = Model(input_size=input_size,
              num_class=num_class,
              batch_size=batch_size,
              num_units=num_units,
              num_layers=num_layers)
data = Data()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    seq_lens = np.ones(batch_size) * (seq_len * word_size)
    for i in range(step):
        inputs, labels = data.next_batch(word_size, batch_size)
        feed = {
            model.inputs : inputs,
            model.target : labels,
            model.seq_len : seq_lens,
            model.keep_prob : 0.5,
            model.lr : learn_rate
        }
        sess.run([model.op], feed_dict=feed)

        if (i+1) % 100 == 0 :
            feed[model.keep_prob] = 1.0
            loss, err, decoded = sess.run([model.loss, model.err, model.decoded], feed_dict=feed)
            ori = data.decode_sparse_tensor(labels)
            pre = data.decode_sparse_tensor(decoded[0])
            acc = data.hit(pre, ori)
            msg = 'train step: %d, accuracy: %.4f, word error: %.6f, loss: %f' % (i+1, acc, err, loss)
            print(msg)
            print('ori: %s\npre: %s' % (ori[0], pre[0]))
