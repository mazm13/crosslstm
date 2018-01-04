import argparse

# dataset
BASE_DIR = './sentdata/'
datasplit = ['train', 'dev', 'test']

# hyperparams
learning_rate = 0.01
weight_decay = 1e-4

input_dim = 128
hidden_dim = 64
output_dim = 32 #no use
pair_dim = 16
num_classes = 5
word_scale = 0.05

use_cuda = True

force_rebuild_vocab = True
display_iter = 500

def parse_args():
    parser = argparse.ArgumentParser(description='CrossLSTM for Sentence Similarity on Parsed Trees')
    
    parser.add_argument('--data', default=BASE_DIR,
                        help='path to dataset')
    parser.add_argument('--glove', default='/home/zoeching/projects/treelstm.pytorch/data/glove/',
                        help='directory with GLOVE embeddings')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in')
    
    parser.add_argument('--epoch', type=int, default=15,
                        help='Training epoch')
    parser.add_argument('--batch_size', type=int, default=25,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--wc', type=float, default=weight_decay,
                        help='weight decay')


    parser.add_argument('--word_scale', type=float, default=word_scale,
                        help='uniform distribution scale of word vector.')
    parser.add_argument('--input_dim', type=int, default=input_dim,
                        help='dim of word vector.')
    parser.add_argument('--hidden_dim', type=int, default=hidden_dim,
                        help="dim of LSTM cell & hidden")
    parser.add_argument('--pair_dim', type=int, default=pair_dim,
                        help="dim of lstm-pair ")
    parser.add_argument('--num_classes', type=int, default=num_classes,
                        help='number of output classes')
    

    parser.add_argument('--use_cuda', default=True,
                        help='enable cuda or not.')

    parser.add_argument('--use_glove', type=bool, default=False,
                        help='enable init from glove embeddings.')
    parser.add_argument('--freeze', type=bool, default=False,
                        help='DO NOT finetune the word embeddings')
    
    parser.add_argument('--weight', type=str, default='',
                        help='weight file for initializing the model.')
    args = parser.parse_args()
    return args
