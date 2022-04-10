# -*- encoding: utf-8 -*-
'''
@File    :   VariantNeedleman-Wunsch.py
@Time    :   2022/02/21 14:45:15
@Author  :   jackfromeast 刘征宇 / NING 王楠楠 
@Version :   2.0
@Contact :   1104120243@qq.com
'''

# here put the import lib
import sys
sys.path.append("..")
import soundshapecode
# import soundshapecode
import numpy as np
import json
import pandas as pd
import pkg_resources

# SSC mode
SSC_ENCODE_WAY = 'ALL' #'ALL','SOUND','SHAPE'


# Variant 对齐
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
            with open(pkg_resources.resource_filename(__name__, '../soundshapecode/zh_data/punc_dict.json'), 'r') as f:
                self.punc_dict = json.loads(f.read())
        
        if uchar in self.punc_dict.keys() or uchar in self.punc_dict.values():
            return True
        else:
            return False

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

    