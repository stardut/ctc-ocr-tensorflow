# -*- coding: utf8 -*-

import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data

class Data(object):
    """docstring for Da"""
    def __init__(self):
        self.mnist = input_data.read_data_sets('data/')

    def next_batch(self, word_size, batch_size):
        imgs = []
        labels = []
        for _ in range(batch_size):
            ims, labs = self.mnist.train.next_batch(word_size)
            ims = np.reshape(ims, (word_size, 28, 28))
            img = np.zeros((28, 28*word_size))
            for i in range(word_size):
                img[:, i*28:(i+1)*28] = ims[i]
            img = np.transpose(img)
            imgs.append(img)            
            labels.append(labs)
        labels = self.sparse_tuple_from(labels)
        return np.asarray(imgs), labels

    def sparse_tuple_from(self, sequences, dtype=np.int32):
        """Create a sparse representention of x.

        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []        
        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)
     
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)     
        return indices, values, shape

    def decode_sparse_tensor(self, sparse_tensor):
        """Transform sparse to sequences ids."""
        decoded_indexes = list()
        current_i = 0
        current_seq = []
        for offset, i_and_index in enumerate(sparse_tensor[0]):
            i = i_and_index[0]
            if i != current_i:
                decoded_indexes.append(current_seq)
                current_i = i
                current_seq = list()
            current_seq.append(offset)
        decoded_indexes.append(current_seq)

        result = []
        for index in decoded_indexes:
            ids = [sparse_tensor[1][m] for m in index]
            text = ''.join(list(map(self.id2word, ids)))
            result.append(text)
        return result

    def hit(self, text1, text2):
        """Calculate accuracy of predictive text and target text."""
        res = []
        for idx, words1 in enumerate(text1):
            res.append(words1 == text2[idx])
        return np.mean(np.asarray(res))

    def id2word(self, idx):
        return str(idx)