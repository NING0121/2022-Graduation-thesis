import os
from pyexpat import model
import torch
import logging

class Config():
    """
    模型及训练参数配置类
    """
    def __init__(self):
        #   数据集设置相关配置
        self.source_dic_path = "./Data/source_vocal.pkl"            # 源语言字典
        self.target_dic_path = "./Data/target_vocal.pkl"            # 目标语言字典
        self.src2tgt_path = "./Data/src_idx2tgt_idx.pkl"            # 源语言和目标语言对应字 字典
        self.train_set_path = "./Data/Dataset/train_data.csv"       # 训练集
        self.test_set_path = "./Data/Dataset/test_data.csv"         # 测试集
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data')
        # self.train_corpus_file_paths = [os.path.join(self.dataset_dir, 'train.de'),   # 训练时编码器的输入
        #                                 os.path.join(self.dataset_dir, 'train.en')]   # 训练时解码器的输入
        # self.val_corpus_file_paths = [os.path.join(self.dataset_dir, 'val.de'),       # 验证时编码器的输入
        #                                 os.path.join(self.dataset_dir, 'val.en')]     # 验证时解码器的输入
        # self.test_corpus_file_paths = [os.path.join(self.dataset_dir, 'test_2016_flickr.de'),
        #                                 os.path.join(self.dataset_dir, 'test_2016_flickr.en')]
        # self.min_freq = 1  # 在构建词表的过程中滤掉词（字）频小于min_freq的词（字）

        # 数据集部分参数
        self.PAD_IDX = 0
        self.BOS_IDX = 1
        self.EOS_IDX = 2
        self.GAP_IDX = 5
        self.source_vocab_size = 5828
        self.target_vocab_size = 3603


        # 训练参数
        self.epochs = 20
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # Transformer 模型训练参数
        self.batch_size = 24
        self.d_model = 128
        self.num_head = 8
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.dim_feedforward = 256
        self.dropout = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 10e-9
        
     


        #  RNNSearch 模型相关配置
        # dropout rate for encoder embedding
        self.enc_emb_dropout = 0.3
        # dropout rate for decoder embedding
        self.dec_emb_dropout = 0.3
        # dropout rate for encoder hidden state
        self.enc_hid_dropout = 0.3
        # dropout rate for readout layer
        self.readout_dropout = 0.3
        # size of source word embedding
        self.enc_ninp = 310
        # size of target word embedding
        self.dec_ninp = 310
        # number of source hidden layer
        self.enc_nhid = 500
        # number of target hidden layer
        self.dec_nhid = 500
        # number of target attention layer
        self.dec_natt = 500
        # number of maxout layer
        self.nreadout = 310



        #  ConvS2S 模型相关配置
        self.embedding_size = 96
        self.out_embedding_size = 256
        self.max_positions = 1024
        self.convolutions = [(96,3)]*9+[(128,3)]*5+[(256,1)]*2
        self.fconv_dropout=0.1
        self.hidden_size = 512
        self.kernel_size = 5
        self.enc_layers = 2
        self.dec_layers = 2
     

        # 模型保存
        # self.model_save_dir = os.path.join(self.project_dir, '../model_params')
        # if not os.path.exists(self.model_save_dir):
        #     os.makedirs(self.model_save_dir)
        # 日志相关
        # self.wandb_Log_name = f'Transformer-CrossEntropyLoss-{self.d_model}_dmodel-{self.num_encoder_layers}_layers-{self.dim_feedforward}_emb-{self.num_head}_head'


        # logger_init(log_file_name='log_train',
        #             log_level=logging.INFO,
        #             log_dir=self.model_save_dir)