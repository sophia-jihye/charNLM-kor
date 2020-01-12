from Preprocessor import Preprocessor
from Quantizer import Quantizer
from model.LSTMCNN import LSTMCNN, load_model
from config import parameters
from utils import *
import pandas as pd
import json

# filepath
kci_korean_json_filepath = parameters.kci_korean_json_filepath
preprocessed_json_filepath = parameters.preprocessed_json_filepath
whole_sentences_txt_filepath = parameters.whole_sentences_txt_filepath
model_param_pkl_filepath = parameters.model_param_pkl_filepath
model_json_filepath = parameters.model_json_filepath
model_weights_h5_filepath = parameters.model_weights_h5_filepath
save_epoch_file = parameters.save_epoch_file

n_words = parameters.n_words
n_chars = parameters.n_chars
char_vec_size = parameters.char_vec_size
feature_maps = parameters.feature_maps
kernels = parameters.kernels
num_layers = parameters.num_layers
batch_size = parameters.batch_size
seq_length = parameters.seq_length
batch_norm = parameters.batch_norm
highway_layers = parameters.highway_layers
num_layers = parameters.num_layers
rnn_size = parameters.rnn_size
dropout = parameters.dropout
learning_rate = parameters.learning_rate
max_grad_norm = parameters.max_grad_norm
max_epochs = parameters.max_epochs
decay_when = parameters.decay_when
learning_rate_decay = parameters.learning_rate_decay
save_every = parameters.save_every

def load_raw_data(filepath):
    with open(filepath,"r") as f:
        d = json.load(f)
    df = pd.read_json(d)
    return df

def raw2preprocessed(preprocessor, df):
    df = preprocessor.remove_outlier_document(df)
    df = preprocessor.remove_outlier_sentence(df)
    
    # json으로 저장
    with open(preprocessed_json_filepath, 'w+') as json_file:
        json.dump(df.to_json(orient='records'), json_file)
        print('Created file:', preprocessed_json_filepath)
    return df

def for_train(preprocessor, df):
    whole_sentences = preprocessor.flatten_whole_sentences(df)
    
    # save as .txt
    f = open(whole_sentences_txt_filepath, 'w')
    for i in range(len(whole_sentences)):
        data = "%s\n" % whole_sentences[i]
        f.write(data)
    f.close()
    print('Created file:', whole_sentences_txt_filepath)
    return whole_sentences

def main():
    df = load_raw_data(kci_korean_json_filepath)
    
    preprocessor = Preprocessor()
    df = raw2preprocessed(preprocessor, df)
    whole_sentences = for_train(preprocessor, df)
    
    loader = Quantizer(whole_sentences)
    word_vocab_size = min(n_words, len(loader.idx2word))
    char_vocab_size = min(n_chars, len(loader.idx2char))
    max_word_l = loader.max_word_l
    print('Word vocab size: %d, Char vocab size: %d, Max word length (incl. padding): %d' % (word_vocab_size, char_vocab_size, max_word_l))

    print('creating an LSTM-CNN with', num_layers, 'layers')
    model = LSTMCNN(char_vocab_size, char_vec_size, feature_maps, kernels, batch_size, seq_length, max_word_l, batch_norm, highway_layers, num_layers, rnn_size, dropout, word_vocab_size, learning_rate, max_grad_norm)
        
    pickle.dump(parameters, open(model_param_pkl_filepath, "wb"))
    model.save(model_json_filepath)
    model.fit_generator(loader.next_batch(Train), loader.split_sizes[Train], max_epochs,
                        loader.next_batch(Validation), loader.split_sizes[Validation], decay_when, learning_rate_decay, save_every, save_epoch_file)
    model.save_weights(model_weights_h5_filepath, overwrite=True)
    
    
if __name__ == '__main__':
    main()