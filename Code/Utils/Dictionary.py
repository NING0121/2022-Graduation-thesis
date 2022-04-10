'''
@File    :   Dictionary.py
@Time    :   2022/03/02 15:52:44
@Author  :   NING 王楠楠 
@Version :   1.0
@Contact :   1104120243@qq.com
'''

# here put the import lib
import re
import pandas as pd
import pickle



# 字典对象，存储 word 和 index 对应
class Dictionary(object):

    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = {}

        self.word2idx = word2idx
        self.idx2word = idx2word


    # 从data中获取
    def load_from_data(self, data_path, usecols):

        self.word2idx = self.__word2idx(data_path, usecols)
        self.idx2word = self.__idx2word(self.word2idx)


    # 构建 word -> index 映射
    def __word2idx(self, data_path, usecols):
        # 预留一些特殊符号
        word2idx = {'[PAD]':0, '[BOS]':1, '[EOS]':2, '[UNK]':3, '[MASK]':4, '[GAP]':5}

        data = pd.read_csv(data_path, usecols=[usecols])
        sentences = data[usecols].tolist()
        word_list = list(set("".join(sentences)))
        for i, w in enumerate(word_list):
            word2idx[w] = i + 6
        
        return word2idx
    

    # 构建 index -> word 映射
    def __idx2word(self, word2idx):

        idx2word = {i : w for i, w in enumerate(word2idx)}
        
        return idx2word


    # 保存到文件
    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)


    # 从文件中加载
    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        dic = cls(word2idx, idx2word)
        dic.word2idx = word2idx
        dic.idx2word = idx2word
        return dic


    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

  
    def __len__(self):
        return len(self.idx2word)

    # @property
    # def ntoken(self):
    #     return len(self.word2idx)

    # @property
    # def padding_idx(self):
    #     return len(self.word2idx)

    # def tokenize(self, sentence, add_word):
    #     sentence = sentence.lower()
    #     sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
    #     words = sentence.split()
    #     tokens = []
    #     if add_word:
    #         for w in words:
    #             tokens.append(self.add_word(w))
    #     else:
    #         for w in words:
    #             tokens.append(self.word2idx[w])
    #     return tokens
    