# -*- encoding: utf-8 -*-
'''
@File    :   VariantNeedleman-Wunsch.py
@Time    :   2022/02/21 14:45:15
@Author  :   jackfromeast 刘征宇 / NING 王楠楠 
@Version :   2.0
@Contact :   1104120243@qq.com
'''
# here put the import lib
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import pkg_resources
import json
import sys
import pandas as pd
import pickle
sys.path.append("/data/NING/2022-Graduation-thesis")
sys.path.append("../..")
sys.path.append("..")
from torch.nn.utils.rnn import pad_sequence
# import soundshapecode
import soundshapecode


# SSC mode
SSC_ENCODE_WAY = 'ALL' #'ALL','SOUND','SHAPE'

# 变体字数据集对象
class VariantWordDataset(torch.utils.data.Dataset):

    def __init__(self, mode, config, isAligned=False, supply_ratio = False):

        self.isAligned = isAligned
        self.supply_ratio = supply_ratio
        self.config = config
        assert mode in {"train","test"}
        
        if mode == "train":
            try:
                data = pd.read_csv(self.config.train_set_path, index_col=0)
                supply_data = pd.read_csv(self.config.train_set_supply_path, index_col=0).sample(frac=supply_ratio, axis=0)
                data = pd.concat([data, supply_data], axis=0, ignore_index=True)
            except:
                print("supply_ratio should set between 0 and 1")
                data = pd.read_csv(self.config.train_set_path, index_col=0)


        else:
            data = pd.read_csv(self.config.test_set_path, index_col=0)

        self.SourceData = data["raw_data"].tolist()
        self.TargetData = data["right_data"].tolist()

        self.source_dic = Dictionary.load_from_file(self.config.source_dic_path)
        self.target_dic = Dictionary.load_from_file(self.config.target_dic_path)

        self.PAD_IDX = self.source_dic.word2idx["[PAD]"]
        self.BOS_IDX = self.source_dic.word2idx["[BOS]"]
        self.EOS_IDX = self.source_dic.word2idx["[EOS]"]
        self.GAP_IDX = self.source_dic.word2idx["[GAP]"]
    
    def __split__():
        pass

    def __len__(self):
        return len(self.SourceData)


    def __getitem__(self, idx):

        if self.isAligned:
            variantNW = VariantNW()
            seq1 = self.SourceData[idx]
            seq2 = self.TargetData[idx]
        

            variantNW.set_seqs(seq1, seq2)
            variantNW.propagate()
            aligned_seq1, aligned_seq2 = variantNW.traceback()

            source = torch.tensor([self.source_dic.word2idx[i] for i in aligned_seq1])
            target = torch.tensor([self.target_dic.word2idx[i] for i in aligned_seq2])

            sample = [source, target]
            
            return sample
        
        else:
            source = torch.tensor([self.source_dic.word2idx[i] for i in self.SourceData[idx]])
            target = torch.tensor([self.target_dic.word2idx[i] for i in self.TargetData[idx]])

            sample = [source, target]

            return sample
    
    def generate_batch(self, data_batch):

        source_batch, target_batch, target_length = [], [], []

        for (source_item, target_item) in data_batch:  # 开始对一个batch中的每一个样本进行处理。

            source_batch.append(source_item)  # 编码器输入序列不需要加起止符

            # 在每个idx序列的首位加上 起始token 和 结束 token
            target = torch.cat([torch.tensor([self.BOS_IDX]), target_item, torch.tensor([self.EOS_IDX])], dim=0)

            target_batch.append(target)
            target_length.append(len(target))

        # 以最长的序列为标准进行填充
        source_batch = pad_sequence(source_batch, padding_value=self.PAD_IDX)  # [de_len,batch_size]

        target_batch = pad_sequence(target_batch, padding_value=self.PAD_IDX)  # [en_len,batch_size]

        target_length =  torch.tensor(target_length, dtype=torch.int64,device="cpu")

        return source_batch, target_batch, target_length, self.isAligned


# Variant 对象
class VariantNW():
    def __init__(self):
        self.seq1 = None
        self.seq2 = None
        
        self.seq1_length = None
        self.seq2_length = None
        self.NWMatrix = None

        # setup score function
        soundshapecode.getHanziStrokesDict()
        soundshapecode.getHanziStructureDict()
        soundshapecode.getHanziSSCDict()

        self.punc_dict = None

    def set_seqs(self, seq1, seq2):
        self.seq1 = ['[GAP]'] + list(seq1)
        self.seq2 = ['[GAP]'] + list(seq2)

        self.seq1_length = len(self.seq1)
        self.seq2_length = len(self.seq2)

        self.__setupMatrix()
    
    def __setupMatrix(self):
        self.NWMatrix = np.zeros((self.seq2_length, self.seq1_length))

        # 对第一行和列进行初始化赋值
        self.NWMatrix[0][0] = 0
        self.NWMatrix[0, 1:] = 0
        self.NWMatrix[1:, 0] = -10

    # 计算NWMatrix中的每一项
    def propagate(self):

        # 按Matrix[layer, layer:]+Maxtrix[layer:, layer]为一层传播遍历
        max_layer = self.seq2_length-1 if self.seq2_length <= self.seq1_length else self.seq1_length-1

        for layer in range(1, max_layer+1):
            self.NWMatrix[layer][layer] = self.__calculate(layer, layer)

            for j in range(layer+1, self.seq1_length):
                self.NWMatrix[layer][j] = self.__calculate(layer, j)

            for i in range(layer+1, self.seq2_length):
                self.NWMatrix[i][layer] = self.__calculate(i, layer)
       
    # 回溯算法
    def traceback(self):
        seq1, seq2 = self.seq1, self.seq2
        n, m = len(seq2) - 1, len(seq1) - 1
        
        def __trace(i, j):
            if i == 0 and j == 0:
                return ''
            if i != 0 and j == 0:
                return __trace(i-1, 0) + '2'
            if i == 0 and j != 0:
                return __trace(0, j-1) + '1'
            if i != 0 and j != 0:
                max_v = max(self.NWMatrix[i, j-1], self.NWMatrix[i-1, j], self.NWMatrix[i-1, j-1])

                # 在回溯时，按照对角、向上、向左的优先规则返回
                if self.NWMatrix[i-1, j-1] == max_v:
                    return __trace(i-1, j-1) + '0'

                if self.NWMatrix[i-1, j] == max_v:
                    return __trace(i-1, j) + '2'

                if self.NWMatrix[i, j-1] == max_v:
                    return __trace(i, j-1) + '1'

        pathcode = __trace(n, m)

        aligned_seq1, aligned_seq2 = [], []
        i, j = 1, 1
        for code in list(pathcode):
            if code == '0':
                aligned_seq1.append(seq1[j])
                j += 1
                aligned_seq2.append(seq2[i])
                i += 1
                
            elif code == '1':
                aligned_seq1.append(seq1[j])
                j += 1
                aligned_seq2.append('[GAP]')
                
            elif code == '2':
                aligned_seq2.append(seq2[i])
                i += 1
                aligned_seq1.append('[GAP]')
                

        return aligned_seq1, aligned_seq2

    # 计算各位置字符相似度评分
    def get_aligned_seq_score(self, seq1, seq2):
        if len(seq1) != len(seq2):
            raise ValueError('Please align the input sequence first!\n')

        score_seq = []
        for i in range(0, len(seq1)):
            score_seq.append(self.__score(seq2[i], seq1[i]))
        
        return score_seq

    def __calculate(self, i, j):
        score1 = self.NWMatrix[i-1, j-1] + self.__score(self.seq2[i], self.seq1[j])   
        score2 = self.NWMatrix[i-1, j] + self.__score(self.seq2[i], '[GAP]')
        score3 = self.NWMatrix[i, j-1] + self.__score('[GAP]', self.seq1[j])

        return max(score1, score2, score3)

    # 字符相似度评分函数
    # char1和char2分别对应来自seq1和seq2
    def __score(self, char2, char1):
        if char1 != '[GAP]' and char2 != '[GAP]':
            # 如果char1与char2均为汉字
            if self.__is_chinese(char1) and self.__is_chinese(char2):
                char1_ccs = soundshapecode.getSSC_char(char1, SSC_ENCODE_WAY)
                char2_ccs = soundshapecode.getSSC_char(char2, SSC_ENCODE_WAY)

                score = soundshapecode.computeSSCSimilaruty(char1_ccs, char2_ccs, SSC_ENCODE_WAY)
                return score * 10 # score = [0,1]
            
            # 如果char1与char2均为字母
            elif self.__is_alphabet(char1) and self.__is_alphabet(char2):
                if char1 == char2:
                    return 10.0

            # 如果char1与char2均为数字
            elif self.__is_number(char1) and self.__is_number(char2):
                if char1 == char2:
                    return 10.0
            
            # 如果char1与char2均为标点符号
            elif self.__is_punc(char1) and self.__is_punc(char2):
                # 默认seq1中的字符可能存在中文标点的情况
                if char1 == char2 or self.punc_dict[char1] == char2:
                    return 10.0

            # 其他错配的情况
            return 0.0

        # 默认情况下char1来源于原文本，char2来源于标注文本
        # 期望标注文本与原文本对齐，故在seq2中添加[GAP]
        elif char1 == '[GAP]' and char2 != '[GAP]':
            return -10.0
        
        elif char1 != '[GAP]' and char2 == '[GAP]':
            return 0.0

    # 判断一个unicode是否是数字
    def __is_number(self, uchar):
        if u'\u0030' <= uchar <= u'\u0039':
            return True
        else:
            return False

    #判断一个unicode是否是英文字母
    def __is_alphabet(self, uchar):
        if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
            return True
        else:
            return False

    # 判断一个unicode是否为汉字
    def __is_chinese(self, uchar):
        if u'\u4e00' <= uchar <= u'\u9fa5':
            return True
        else:
            return False

    def __is_punc(self, uchar):
        if self.punc_dict == None:
            with open(pkg_resources.resource_filename(__name__, './soundshapecode/zh_data/punc_dict.json'), 'r') as f:
                self.punc_dict = json.loads(f.read())
        
        if uchar in self.punc_dict.keys() or uchar in self.punc_dict.values():
            return True
        else:
            return False


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




# if __name__== "__main__":
#     variantNW = VariantNW()
    
#     data = pd.read_csv("Data/data.csv", nrows=5)

#     raw_data = data["raw_data"].tolist()
#     right_data = data["right_data"].tolist()

#     length = len(raw_data)


#     aligned_seq1_list = []
#     aligned_seq2_list = []

#     for i in range(length):
#         seq1 = raw_data[i]
#         seq2 = right_data[i]
    

#         variantNW.set_seqs(seq1, seq2)
#         variantNW.propagate()

#         # print(variantNW.NWMatrix)
#         aligned_seq1, aligned_seq2 = variantNW.traceback()


#         aligned_seq1_list.append(aligned_seq1)
#         aligned_seq2_list.append(aligned_seq2)
    

#     print(len(aligned_seq1_list))
#     print(aligned_seq1_list[0:3])
#     print(aligned_seq2_list[0:3])



        # print(variantNW.get_aligned_seq_score(aligned_seq1, aligned_seq2))
    
    # seq1 = "最新【辐莉"
    # seq2 = "最新福利"

    # variantNW.set_seqs(seq1, seq2)
    # variantNW.propagate()
    # print(variantNW.NWMatrix)
    # aligned_seq1, aligned_seq2 = variantNW.traceback()
    # print(aligned_seq1)
    # print(aligned_seq2)
    # print(variantNW.get_aligned_seq_score(aligned_seq1, aligned_seq2))