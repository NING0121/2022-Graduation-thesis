from distutils.command.config import config
from numpy import int64
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """"encode the input sequence with Bi-GRU"""
    def __init__(self, ninp, nhid, ntok, padding_idx, emb_dropout, hid_dropout):
        super(Encoder, self).__init__()
        self.nhid = nhid
        self.emb = nn.Embedding(ntok, ninp, padding_idx=padding_idx)
        self.bi_gru = nn.GRU(ninp, nhid, 1, batch_first=True, bidirectional=True)
        self.enc_emb_dp = nn.Dropout(emb_dropout)
        self.enc_hid_dp = nn.Dropout(hid_dropout)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        h0 = weight.new_zeros(2, batch_size, self.nhid)
        return h0

    def forward(self, input, mask):
        """
            对原始数据进行编码encoder
            :param input:     [batch_size, src_len]
            :param mask:   [batch_size, src_len]
            :output output [batch_size, src_len]
        """
        hidden = self.init_hidden(input.size(0))
        #self.bi_gru.flatten_parameters()
        input = self.enc_emb_dp(self.emb(input))
        # 计算有效字符长度
        length = mask.sum(1).tolist()
        total_length = mask.size(1)
        input = torch.nn.utils.rnn.pack_padded_sequence(input, length, batch_first=True, enforce_sorted=False)
        output, hidden = self.bi_gru(input, hidden)
        output = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=total_length)[0]
        output = self.enc_hid_dp(output)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        # print(f"encoder:{output.shape}")
        return output, hidden


class Attention(nn.Module):
    """Attention Mechanism"""
    def __init__(self, nhid, ncontext, natt):
        super(Attention, self).__init__()
        self.h2s = nn.Linear(nhid, natt)
        self.s2s = nn.Linear(ncontext, natt)
        self.a2o = nn.Linear(natt, 1)

    def forward(self, hidden, mask, context):
        shape = context.size()
        attn_h = self.s2s(context.view(-1, shape[2]))
        attn_h = attn_h.view(shape[0], shape[1], -1)
        attn_h += self.h2s(hidden).unsqueeze(1).expand_as(attn_h)
        logit = self.a2o(F.tanh(attn_h)).view(shape[0], shape[1])
        if mask.any():
            logit.data.masked_fill_(1 - mask, -float('inf'))
        softmax = F.softmax(logit, dim=1)
        output = torch.bmm(softmax.unsqueeze(1), context).squeeze(1)
        return output


class VallinaDecoder(nn.Module):
    def __init__(self, ninp, nhid, enc_ncontext, natt, nreadout, readout_dropout):
        super(VallinaDecoder, self).__init__()
        self.gru1 = nn.GRUCell(ninp, nhid)
        self.gru2 = nn.GRUCell(enc_ncontext, nhid)
        self.enc_attn = Attention(nhid, enc_ncontext, natt)
        self.e2o = nn.Linear(ninp, nreadout)
        self.h2o = nn.Linear(nhid, nreadout)
        self.c2o = nn.Linear(enc_ncontext, nreadout)
        self.readout_dp = nn.Dropout(readout_dropout)

    def forward(self, emb, hidden, enc_mask, enc_context):
        hidden = self.gru1(emb, hidden)
        attn_enc = self.enc_attn(hidden, enc_mask, enc_context)
        hidden = self.gru2(attn_enc, hidden)
        output = F.tanh(self.e2o(emb) + self.h2o(hidden) + self.c2o(attn_enc))
        output = self.readout_dp(output)

        # print(enc_mask.shape)
        # print(enc_context.shape)
        # print(output.shape)

        return output, hidden


class RNNSearch(nn.Module):
    def __init__(self, config):
        super(RNNSearch, self).__init__()

        self.dec_nhid = config.dec_nhid
        self.BOS_IDX = config.BOS_IDX
        self.EOS_IDX = config.EOS_IDX
        self.PAD_IDX = config.PAD_IDX
        self.PAD_IDX = config.PAD_IDX
        

        self.emb = nn.Embedding(config.target_vocab_size, config.dec_ninp, padding_idx=config.PAD_IDX)
        self.encoder = Encoder(config.enc_ninp, config.enc_nhid, config.source_vocab_size, config.PAD_IDX, config.enc_emb_dropout, config.enc_hid_dropout)
        self.decoder = VallinaDecoder(config.dec_ninp, config.dec_nhid, 2 * config.enc_nhid, config.dec_natt, config.nreadout, config.readout_dropout)
        self.affine = nn.Linear(config.nreadout, config.target_vocab_size)
        
        self.init_affine = nn.Linear(2 * config.enc_nhid, config.dec_nhid)
        self.dec_emb_dp = nn.Dropout(config.dec_emb_dropout)

    def forward(self, src, src_mask, f_trg, f_trg_mask, b_trg=None, b_trg_mask=None):
        enc_context, _ = self.encoder(src, src_mask)

        batch_size = src.shape[0]

        # 使内存连续
        enc_context = enc_context.contiguous()

        # 对句子求和
        avg_enc_context = enc_context.sum(1)

        # 求和，并使其和句子形状相同
        enc_context_len = src_mask.sum(1).unsqueeze(-1).expand_as(avg_enc_context)

        # 
        avg_enc_context = avg_enc_context / enc_context_len

        attn_mask = src_mask.byte()

        hidden = F.tanh(self.init_affine(avg_enc_context))

        loss = 0

        # 生成0向量用于拼接后续向量
        outputs = torch.zeros(batch_size,1, device="cuda", dtype=torch.int64)

        for i in range(f_trg.size(1) - 1):
            
            

            output, hidden = self.decoder(self.dec_emb_dp(self.emb(f_trg[:, i])), hidden, attn_mask, enc_context)
            loss += F.cross_entropy(self.affine(output), f_trg[:, i+1], reduce=False) * f_trg_mask[:, i+1]

            # 预测值
            predict = self.affine(output).max(1,keepdim=True)[1]
            # predict = self.affine(output)
            outputs = torch.cat((outputs,predict), 1)
        
        # output为模型输出
        # print(outputs.shape)


        # 不同句子的loss求和 / target 字数目
        w_loss = loss.sum() / f_trg_mask[:, 1:].sum()

        # 每个字的loss
        loss = loss.mean()
        return loss.unsqueeze(0), w_loss.unsqueeze(0), outputs
