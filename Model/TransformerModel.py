from re import X
import sys
sys.path.append("../")
import torch
from torch import Tensor, logit, nn
import torch.nn.functional as F
import pytorch_lightning as pl
from Networks.TranslationModel import TranslationModel
from data_loader import Dictionary
from torchtext.data.metrics import bleu_score
import numpy as np
import pickle
from config import PAD_IDX, GAP_IDX, SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE

class TransformerModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.check_name = f"TransformerModel"
        self.config = config
        self.log_name = f"TransformerModel-{self.config.d_model}_dmodel-{self.config.num_encoder_layers}_layers-{self.config.dim_feedforward}_emb-{self.config.num_head}_head"


        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        self.model = TranslationModel(src_vocab_size=SOURCE_VOCAB_SIZE,
                                         tgt_vocab_size=TARGET_VOCAB_SIZE,
                                         d_model=config.d_model,
                                         nhead=config.num_head,
                                         num_encoder_layers=config.num_encoder_layers,
                                         num_decoder_layers=config.num_decoder_layers,
                                         dim_feedforward=config.dim_feedforward,
                                         dropout=config.dropout)
        
        self.source_dict = Dictionary.load_from_file(config.source_dic_path)
        self.target_dict = Dictionary.load_from_file(config.target_dic_path)

        with open(self.config.src2tgt_path, 'rb') as fp:
            self.src2tgt = pickle.load(fp)

    def forward(self, src=None, tgt=None, src_mask=None,
                tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        logits = self.model.forward(src=src, tgt=tgt, src_mask=src_mask,
                tgt_mask=tgt_mask, memory_mask=memory_mask, src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            
        return logits


    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    
    def confusion_matrix(self, src, logits, tgt_out, PAD_IDX):
        """
            给定变体字序列、生成序列以及标签序列，生成混淆矩阵
            :param src:     [src_len, batch_size]
            :param logits:  [tgt_len,batch_size,tgt_vocab_size]
            :param y_true:  [tgt_len,batch_size]
            :param PAD_IDX:
        """
        FP = 0 # 变体字还原错误或正常文字被错误还原
        FN = 0 # 变体字漏检计
        TP = 0 # 变体字还原结果正确

        y_pred = logits[:-1,:,:].transpose(0, 1).argmax(axis=2).reshape(-1)
        y_ture = tgt_out[:-1,:].transpose(0, 1).reshape(-1)
        x_origin = src.transpose(0, 1).reshape(-1)

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

    def BLEU_score(self, source, target):

        # 预测值
        y_pred = source[:-1,:,:].transpose(0, 1).argmax(axis=2).reshape(-1)
        # 真实值
        y_ture = target[:-1,:].transpose(0, 1).reshape(-1)

        y_pred = np.delete(Tensor.cpu(y_pred).numpy() , np.where(Tensor.cpu(y_ture).numpy() <= GAP_IDX))
        y_ture = np.delete(Tensor.cpu(y_ture).numpy() , np.where(Tensor.cpu(y_ture).numpy() <= GAP_IDX))

        candidate = [str(i) for i in y_pred.tolist()]
        reference = [str(i) for i in y_ture.tolist()]

        return bleu_score([candidate], [[reference]])


    def shared_step(self, batch, stage):
        src, tgt, tgt_length, isAligned = batch

        src = src.to(self.device)  # [src_len, batch_size]
        tgt = tgt.to(self.device)


        tgt_input = tgt[:-1, :]  # 解码部分的输入, [tgt_len,batch_size]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask \
                = self.create_mask(src, tgt_input)
        
        # logits 输出shape为[tgt_len,batch_size,tgt_vocab_size]
        logits = self.forward(
                                src=src,  # Encoder的token序列输入，[src_len,batch_size]
                                tgt=tgt_input,  # Decoder的token序列输入,[tgt_len,batch_size]
                                src_mask=src_mask,  # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的
                                tgt_mask=tgt_mask, # Decoder的注意力Mask输入，用于掩盖当前position之后的position [tgt_len,tgt_len]
                                src_key_padding_mask=src_padding_mask,  # 用于mask掉Encoder的Token序列中的padding部分
                                tgt_key_padding_mask=tgt_padding_mask,  # 用于mask掉Decoder的Token序列中的padding部分
                                memory_key_padding_mask=src_padding_mask)  # 用于mask掉Encoder的Token序列中的padding部分
    

        ### 计算loss
        tgt_out = tgt[1:, :]  # 解码部分的真实值  shape: [tgt_len,batch_size]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        # [tgt_len*batch_size, tgt_vocab_size] with [tgt_len*batch_size, ]

        BLEU_SCORE = self.BLEU_score(logits, tgt_out)
        result = {"loss": loss,
                       "BLEU_SCORE":BLEU_SCORE}
        
        if isAligned:
            TP, FP, FN = self.confusion_matrix(src, logits, tgt_out, PAD_IDX)
            result["TP"] = TP
            result["FP"] = FP
            result["FN"] = FN
        
        return result





    def shared_epoch_end(self, outputs, stage):

        loss = torch.mean(torch.FloatTensor([i["loss"] for i in outputs]))
        BLEU_SCORE = torch.Tensor([x["BLEU_SCORE"] for x in outputs]).to('cpu').mean().item()
        metrics = {
            f"{stage}_loss": loss.item(),
            f"{stage}_BLEU_SCORE": BLEU_SCORE,}
        
        if "TP" in outputs[0].keys():
            tp = torch.Tensor([x["TP"] for x in outputs]).to('cpu').sum().item()
            fn = torch.Tensor([x["FN"] for x in outputs]).to('cpu').sum().item()
            fp = torch.Tensor([x["FP"] for x in outputs]).to('cpu').sum().item()
            precision = self.precision_score(tp, fp)
            recall = self.recall_score(tp, fn)
            f1 = self.f1_score(precision, recall)
            metrics[ f"{stage}_precison"] = precision
            metrics[ f"{stage}_recall"] = recall
            metrics[ f"{stage}_f1"] = f1

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
        # return torch.optim.Adam(self.parameters(), lr=1e-3)
        # weight_decay = 1e-6  # l2正则化系数
        # 假如有两个网络，一个encoder一个decoder
        # optimizer = torch.optim.Adam([{'encoder_params': self.encoder.parameters()}, {'decoder_params': self.decoder.parameters()}], lr=1e-3, weight_decay=weight_decay)


        # 同样，如果只有一个网络结构，就可以更直接了
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # 我这里设置 2,4 个epoch后学习率变为原来的0.5，之后不再改变

        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15,20,25], gamma=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict