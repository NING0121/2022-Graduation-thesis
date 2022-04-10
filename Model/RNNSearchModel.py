import sys
from turtle import backward
sys.path.append("../")
from distutils.command.config import config
import os
from cv2 import mean
from sklearn.metrics import precision_recall_curve
import torch
from torch import logit, nn, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.autograd import Variable
from Model_parts.RNNSearch import RNNSearch
from Code.Utils.config import Config
from Code.Utils import Config, Dictionary
from torchtext.data.metrics import bleu_score
import numpy as np
import pickle

class RNNSearchModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.check_name = f"RNNsearchModel-CrossEntropyLoss"
        self.config = config
        self.log_name = f"RNNsearchModel-CrossEntropyLoss-{self.config.enc_ninp}_ninp-{self.config.enc_nhid}_nhid-{self.config.dec_natt}_natt-{self.config.enc_emb_dropout}_drop"

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.PAD_IDX)

        self.model = RNNSearch(self.config)
        
        self.source_dict = Dictionary.load_from_file(config.source_dic_path)
        self.target_dict = Dictionary.load_from_file(config.target_dic_path)

        with open(self.config.src2tgt_path, 'rb') as fp:
            self.src2tgt = pickle.load(fp)

    def forward(self, src, src_mask, f_trg, f_trg_mask, b_trg=None, b_trg_mask=None):
        
        logits = self.model.forward(src = src, src_mask = src_mask, f_trg = f_trg, f_trg_mask = f_trg_mask, b_trg=None, b_trg_mask=None)

        return logits


    def create_mask(self, vector):

        vector_mask = (vector != self.config.PAD_IDX)

        return vector_mask
    
    def traverse(self, tensor):

        return tensor.flip([0])
    
    def confusion_matrix(self, src, output, y_ture, PAD_IDX):
        """
            给定变体字序列、生成序列以及标签序列，生成混淆矩阵
            :param src:     [batch_size, src_len]
            :param output:  [batch_size, tgt_len]
            :param y_true:  [batch_size, tgt_len]
            :param PAD_IDX:
        """
        FP = 0 # 变体字还原错误或正常文字被错误还原
        FN = 0 # 变体字漏检计
        TP = 0 # 变体字还原结果正确

        
        y_pred = output[:,1:-1].reshape(-1)
        y_ture = y_ture[:,1:-1].reshape(-1)
        x_origin = src.reshape(-1)

        # 所有mask的位置
        padd_pos = torch.where(x_origin==PAD_IDX, False, True)
        
        # 首先筛选出不在src2tgt字典中的src索引为变体字
        x_origin = x_origin.to('cpu')

        def first_variant_select(idx): return True if idx not in self.src2tgt.keys() else False
        first_variant_select = np.vectorize(first_variant_select)
        is_variant = first_variant_select(x_origin.numpy())

        # 进一步筛选出来两序列对应字符不相同的变体字
        def idx_transform(idx): return self.src2tgt[idx] if idx in self.src2tgt.keys() else idx
        idx_transform = np.vectorize(idx_transform)
        x_target = idx_transform(x_origin.numpy()) # 由target字典索引表示的原始变体字序列，其中不在src2tgt字典中的索引保留原始的索引即可(已在上一步处理)

        x_target = torch.from_numpy(x_target).to(self.device)
        x_origin = x_origin.to(self.device)
        is_variant = torch.from_numpy(is_variant).to(self.device)
        is_variant = torch.logical_or(~torch.eq(x_target, y_ture), is_variant)

        non_variant = torch.logical_and(~is_variant, padd_pos) # 非变体字且非PAD
        is_variant = torch.logical_and(is_variant, padd_pos) # 是变体字且非PAD

        # 计算FP
        # 正常文字被错误还原
        FP = torch.sum((y_pred[non_variant] != y_ture[non_variant]).int())
        # 或者变体字还原错误
        FP += torch.sum((y_pred[is_variant] != y_ture[is_variant]).int())

        # 计算TN
        # 变体字漏检
        FN = torch.sum((y_pred[is_variant] == x_target[is_variant]).int())

        # 计算TP
        # 变体字被正确还原
        TP = torch.sum((y_pred[is_variant] == y_ture[is_variant]).int())

        return TP, FP, FN
    
    def f1_score(self, precision, recall):
        if precision == 0.0 and recall == 0.0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision+recall)
        return f1
    
    def precision_score(self, tp, fp):
        # 防止除0
        if tp == 0.0 and fp == 0.0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        return precision

    def recall_score(self, tp, fn):
        # 防止除0
        if tp == 0.0 and fn == 0.0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        
        return recall

    def BLEU_score(self, decoder_out, y_ture):

        # 预测值
        y_pred = decoder_out[:,1:-1].reshape(-1)


        # 真实值
        y_ture = y_ture[:,1:-1].reshape(-1)

        y_pred = np.delete(Tensor.cpu(y_pred).numpy() , np.where(Tensor.cpu(y_pred).numpy() <= self.config.GAP_IDX))
        y_ture = np.delete(Tensor.cpu(y_ture).numpy() , np.where(Tensor.cpu(y_ture).numpy() <= self.config.GAP_IDX))

        candidate = [str(i) for i in y_pred.tolist()]
        reference = [str(i) for i in y_ture.tolist()]

        return bleu_score([candidate], [[reference]])


    def shared_step(self, batch, stage):
        src, tgt = batch

        src = src.to(self.device).transpose(0, 1) # [batch_size, src_len]
        tgt = tgt.to(self.device).transpose(0, 1)

        forward_tgt = tgt
        backward_tgt = self.traverse(tgt)

        src_mask = self.create_mask(src)
        forward_tgt_mask = self.create_mask(forward_tgt)
        backward_tgt_mask = self.create_mask(backward_tgt)

        
        # logits 输出shape为[tgt_len,batch_size,tgt_vocab_size]
        loss, w_loss, output = self.forward(
            src = src,                   # Encoder的token序列输入，[src_len,batch_size]
            src_mask = src_mask, 
            f_trg = forward_tgt, 
            f_trg_mask = forward_tgt_mask,
            b_trg=backward_tgt, 
            b_trg_mask=backward_tgt_mask)
        

        # TP, FP, FN = self.confusion_matrix(src, output, forward_tgt, self.config.PAD_IDX)
        BLEU_SCORE = self.BLEU_score(output, forward_tgt)

        return {"loss": loss.mean(),
                # "TP": TP,
                # "FN": FN,
                # "FP": FP, 
                "BLEU_SCORE":BLEU_SCORE
                }




    def shared_epoch_end(self, outputs, stage):
        loss = torch.mean(torch.FloatTensor([i["loss"] for i in outputs]))
        # tp = torch.Tensor([x["TP"] for x in outputs]).to('cpu').sum().item()
        # fn = torch.Tensor([x["FN"] for x in outputs]).to('cpu').sum().item()
        # fp = torch.Tensor([x["FP"] for x in outputs]).to('cpu').sum().item()
        BLEU_SCORE = torch.Tensor([x["BLEU_SCORE"] for x in outputs]).to('cpu').mean().item()

        # precision = self.precision_score(tp, fp)
        # recall = self.recall_score(tp, fn)
        # f1 = self.f1_score(precision, recall)

        metrics = {
            f"{stage}_loss": loss.item(),
            # f"{stage}_precison": precision,
            # f"{stage}_recall": recall,
            # f"{stage}_f1": f1,
            f"{stage}_BLEU_SCORE": BLEU_SCORE,
        }
        
        self.log_dict(metrics, prog_bar=True)
        self.logger.log_metrics(metrics)





    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.model.parameters(),
        #                          lr=0.,
        #                          betas=(self.config.beta1, self.config.beta2), eps=self.config.epsilon)

        return torch.optim.Adam(self.parameters(), lr=5e-4)