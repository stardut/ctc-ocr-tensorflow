# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from model import Model

from data import Data

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_size = 28
num_class = 10 + 1
num_layers = 2
seq_len = 28
batch_size = 64
num_units = 256
word_size = 8
learn_rate = 0.001
step = 100000

model_dir = 'model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model = Model(input_size=input_size,
              num_class=num_class,
              batch_size=batch_size,
              num_units=num_units,
              num_layers=num_layers)
data = Data()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
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
        sess.run(model.op, feed_dict=feed)

        if (i+1) % 100 == 0 or i == step-1:
            feed[model.keep_prob] = 1.0
            loss, err, decode = sess.run([model.loss, model.err, model.decoded], feed_dict=feed)
            ori = data.decode_sparse_tensor(labels)
            pre = data.decode_sparse_tensor(decode[0])
            acc = data.hit(pre, ori)
            msg = 'train step: %d, accuracy: %.4f, word error: %.6f, loss: %f, lr: %f' % (i+1, acc, err, loss, learn_rate)
            print(msg)
            print('ori: %s\npre: %s' % (ori[0], pre[0]))

        if (i+1) % 1000 == 0 or i == step-1:
            checkpoint_path = os.path.join(model_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=i)
            print('save model: ' + checkpoint_path)
            learn_rate = max(0.000001, learn_rate * 0.98)