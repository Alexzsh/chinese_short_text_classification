# coding: utf-8

from __future__ import print_function

import os
import sys
import tensorflow as tf
import tensorflow.contrib.keras as kr

from nn_model import TNNConfig, TextNN
from data.cnews_loader import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data'
vocab_dir = os.path.join(base_dir, 'data.vocab.txt')

save_dir = 'checkpoints'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class nnModel:
    def __init__(self,nn):
        self.config = TNNConfig(nn)
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ['cnn','rnn'] :
        raise ValueError("""usage: python pre_nn.py [cnn / rnn] [content]""")
    save_dir=os.path.join(save_dir,sys.argv[1])
    nn_model = nnModel(sys.argv[1])
    print(nn_model.predict(str(sys.argv[2])))