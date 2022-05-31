import argparse
from torch import cuda

# 数据集使用常量
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
GAP_IDX = 5
SOURCE_VOCAB_SIZE = 5828
TARGET_VOCAB_SIZE = 3603


def arg_parse():
    parser = argparse.ArgumentParser()

    """
    Be careful with the following configurations: 
    --model
    """
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--train', action='store_true', default=True, help='Train model')


    # Dataset Path
    parser.add_argument('--source_dic_path', default="./Data/source_vocal.pkl"  )
    parser.add_argument('--target_dic_path', default="./Data/target_vocal.pkl")
    parser.add_argument('--src2tgt_path', default="./Data/src_idx2tgt_idx.pkl"  )
    parser.add_argument('--train_set_path', default="./Data/Dataset/train_data.csv")
    parser.add_argument('--train_set_supply_path', default="./Data/Dataset/train_data_supply_0417.csv")
    parser.add_argument('--test_set_path', default="./Data/Dataset/test_data.csv")


    # Train
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--epochs', default=35, type=int)
    parser.add_argument('--train_val_ratio', default=0.25)
    parser.add_argument('--lr', default=1e-5, help='')
    parser.add_argument('--batch_size', default=24, type=int, help='')
    parser.add_argument('--device', default="cuda:0" if cuda.is_available() else "cpu", type=str, help='')
    parser.add_argument('--isAligned', default=False, type=bool, help='Whether to enable text alignment mechanism')
    parser.add_argument('--supply_ratio',  default=1, type=float, help='The ratio of the enhanced data to the original data')

    # Checkpoint
    parser.add_argument('--checkpoint_path', default='./Weights')
    parser.add_argument('--checkpoint_moniter_metirc', default='valid_avg_f1')
    parser.add_argument('--checkpoint_filename', default='ConvS2S-{epoch:02d}-{valid_avg_f1:.2f}')


    # Logger
    parser.add_argument('--use_logger', action='store_true', default=False, help='Whether to use logger(wandb)')
    parser.add_argument('--logs_path', default='./Logs')
    parser.add_argument('--logger_filename', default='')



    # Model
    parser.add_argument('--model', default='ConvS2S', choices = ['RNNSearch', 'ConvS2S', 'Transformer'], help='Select a Model first.')
    # RNNSearch
    parser.add_argument('--enc_emb_dropout', default=0.3, help='encoder embedding dropout')
    parser.add_argument('--dec_emb_dropout', default=0.3, help='decoder embedding dropout')
    parser.add_argument('--enc_hid_dropout', default=0.3, help='encoder hidden dropout')
    parser.add_argument('--readout_dropout', default=0.3 , help='readout dropout')
    parser.add_argument('--enc_ninp', default=310, type=int, help='encoder input size')
    parser.add_argument('--dec_ninp', default=310, type=int, help='decoder input size')
    parser.add_argument('--enc_nhid', default=500, type=int, help='encoder hidden size')
    parser.add_argument('--dec_nhid', default=500, type=int, help='decoder hidden size')
    parser.add_argument('--dec_natt', default=500, type=int, help='decoder attention size')
    parser.add_argument('--nreadout', default=310, type=int, help='number of readout')
    # Transformer
    parser.add_argument('--d_model', default=128, help='dimension of model')
    parser.add_argument('--num_head', default=4, help='number of head')
    parser.add_argument('--num_encoder_layers', default=3, help='number of encoder layers')
    parser.add_argument('--num_decoder_layers', default=3 , help='number of decoder layers')
    parser.add_argument('--dim_feedforward', default=256, type=int, help='dimension of feedforward')
    parser.add_argument('--dropout', default=0.1, type=int, help='dropout')
    parser.add_argument('--beta1', default=0.9, type=int, help='beta1')
    parser.add_argument('--beta2', default=0.98, type=int, help='beta2')
    parser.add_argument('--epsilon', default=10e-9, type=int, help='epsilon')
    # ConvS2S
    parser.add_argument('--embedding_size', default=96, help='embedding size')
    parser.add_argument('--out_embedding_size', default=256, help='output embedding size')
    parser.add_argument('--max_positions', default=1024, help='max positions')
    parser.add_argument('--convolutions', default= [(96,3)]*9+[(128,3)]*5+[(256,1)]*2, help='convolutions')
    parser.add_argument('--fconv_dropout', default=0.1, type=int, help='fconv dropout')
    parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    parser.add_argument('--kernel_size', default=5, type=int, help='kernel size')
    parser.add_argument('--enc_layers', default=2, type=int, help='encoder layers')
    parser.add_argument('--dec_layers', default=2, type=int, help='decoder layers')


    args = parser.parse_args()

    return args