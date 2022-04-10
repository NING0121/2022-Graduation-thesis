import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.path.append("H:\项目管理\毕业设计\project")
sys.path.append("../..")
from Utils.VariantNeedleman_Wunsch import VariantNW
from Utils.Dictionary import Dictionary
from torch.nn.utils.rnn import pad_sequence
from Code.Utils.config import Config

config = Config()

class VariantWordDataset(torch.utils.data.Dataset):


    def __init__(self, mode, s_dic_path, t_dic_path, isAligned=False):
        self.isAligned = isAligned

        assert mode in {"train","test"}
        
        if mode == "train":
            data = pd.read_csv(config.train_set_path, index_col=0)

        else:
            data = pd.read_csv(config.test_set_path, index_col=0)

        self.SourceData = data["raw_data"].tolist()
        self.TargetData = data["right_data"].tolist()

        self.source_dic = Dictionary.load_from_file(s_dic_path)
        self.target_dic = Dictionary.load_from_file(t_dic_path)

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

        source_batch, target_batch = [], []

        for (source_item, target_item) in data_batch:  # 开始对一个batch中的每一个样本进行处理。

            source_batch.append(source_item)  # 编码器输入序列不需要加起止符

            # 在每个idx序列的首位加上 起始token 和 结束 token
            target = torch.cat([torch.tensor([self.BOS_IDX]), target_item, torch.tensor([self.EOS_IDX])], dim=0)

            target_batch.append(target)

        # 以最长的序列为标准进行填充
        source_batch = pad_sequence(source_batch, padding_value=self.PAD_IDX)  # [de_len,batch_size]

        target_batch = pad_sequence(target_batch, padding_value=self.PAD_IDX)  # [en_len,batch_size]

        return source_batch, target_batch