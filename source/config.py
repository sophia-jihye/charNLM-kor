from utils import *
import os
import argparse

parser = argparse.ArgumentParser(description='Train a word+character-level language model')
parser.add_argument('--batch_size', type=int, default=20, help='number of sequences to train on in parallel')
parser.add_argument('--seq_length', type=int, default=35, help='number of timesteps to unroll for')
parser.add_argument('--max_word_l', type=int, default=100, help='maximum word length')
parser.add_argument('--n_words', type=int, default=45000, help='max number of words in model')
parser.add_argument('--n_chars', type=int, default=100, help='max number of char in model')
parser.add_argument('--char_vec_size', type=int, default=15, help='dimensionality of character embeddings')
parser.add_argument('--feature_maps', type=int, nargs='+', default=[50,100,150,200,200,200,200], help='number of feature maps in the CNN')
parser.add_argument('--kernels', type=int, nargs='+', default=[1,2,3,4,5,6,7], help='conv net kernel widths')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the LSTM')
parser.add_argument('--batch_norm', type=int, default=0, help='use batch normalization over input embeddings (1=yes)')
parser.add_argument('--highway_layers', type=int, default=2, help='number of highway layers')
parser.add_argument('--rnn_size', type=int, default=650, help='size of LSTM internal state')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout. 0 = no dropout')
parser.add_argument('--learning_rate', type=float, default=1, help='starting learning rate')
parser.add_argument('--max_grad_norm', type=float, default=5, help='normalize gradients at')
parser.add_argument('--max_epochs', type=int, default=25, help='number of full passes through the training data')
parser.add_argument('--decay_when', type=float, default=1, help='decay if validation perplexity does not improve by more than this much')
parser.add_argument('--learning_rate_decay', type=float, default=0.5, help='learning rate decay')
parser.add_argument('--save_every', type=int, default=5, help='save every n epochs')
args = parser.parse_args()

# User configuration
kci_korean_document_length_outlier_short = 60
kci_korean_sentence_length_outlier_short = 35
kci_korean_semtemce_length_outlier_long = 8000
min_df = 0.95
remove_len = False

# System configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('base_dir:', base_dir)
data_dir = os.path.join(base_dir, 'data')
output_base_dir = os.path.join(base_dir, 'output')
now_time_str_dir = os.path.join(base_dir, 'output', now_time_str())
create_dirs([output_base_dir, now_time_str_dir])

# filepath
kci_korean_json_filepath = os.path.join(data_dir, 'kci_korean_sentences_510_191230.json')
log_dir = os.path.join(now_time_str_dir)
preprocessed_json_filepath = os.path.join(now_time_str_dir, 'outlier_removed.json')
whole_units_for_train_txt_filepath = os.path.join(now_time_str_dir, 'whole_units_for_train.txt')
vocab_filepath = os.path.join(now_time_str_dir, 'vocab.npz')
tensor_file = os.path.join(now_time_str_dir, 'data')
char_file = os.path.join(now_time_str_dir, 'data_char')
model_param_pkl_filepath = os.path.join(now_time_str_dir, 'model_param.pkl')
model_json_filepath = os.path.join(now_time_str_dir, 'model.json')
model_weights_h5_filepath = os.path.join(now_time_str_dir, 'model_weights.h5')
save_epoch_file = os.path.join(now_time_str_dir, 'epoch')

class Parameters:
    def __init__(self):
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.max_word_l = args.max_word_l
        self.n_words = args.n_words
        self.n_chars = args.n_chars
        self.char_vec_size = args.char_vec_size
        self.feature_maps = args.feature_maps
        self.kernels = args.kernels
        self.num_layers = args.num_layers
        self.batch_norm = args.batch_norm
        self.highway_layers = args.highway_layers
        self.num_layers = args.num_layers
        self.rnn_size = args.rnn_size
        self.dropout = args.dropout
        self.learning_rate = args.learning_rate
        self.max_grad_norm = args.max_grad_norm
        self.max_epochs = args.max_epochs
        self.decay_when = args.decay_when
        self.learning_rate_decay = args.learning_rate_decay
        self.save_every = args.save_every
        
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.output_base_dir = output_base_dir
        self.kci_korean_document_length_outlier_short = kci_korean_document_length_outlier_short
        self.kci_korean_sentence_length_outlier_short = kci_korean_sentence_length_outlier_short
        self.kci_korean_semtemce_length_outlier_long = kci_korean_semtemce_length_outlier_long
        self.min_df = min_df
        self.remove_len = remove_len
        
        self.kci_korean_json_filepath = kci_korean_json_filepath
        self.log_dir = log_dir
        self.preprocessed_json_filepath = preprocessed_json_filepath
        self.whole_units_for_train_txt_filepath = whole_units_for_train_txt_filepath
        self.vocab_filepath = vocab_filepath
        self.tensor_file = tensor_file
        self.char_file = char_file
        self.model_param_pkl_filepath = model_param_pkl_filepath
        self.model_json_filepath = model_json_filepath
        self.model_weights_h5_filepath = model_weights_h5_filepath
        self.save_epoch_file = save_epoch_file

    def __str__(self):
        item_strf = ['{} = {}'.format(attribute, value) for attribute, value in self.__dict__.items()]
        strf = 'Parameters(\n  {}\n)'.format('\n  '.join(item_strf))
        return strf


parameters = Parameters()
print(parameters)