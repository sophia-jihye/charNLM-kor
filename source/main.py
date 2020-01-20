from Preprocessor import Preprocessor
from Quantizer import Quantizer
from LSTMCNN import LSTMCNN, load_model
from config import parameters
from utils import *
import pandas as pd
import json
import pickle

# filepath
log_dir = parameters.log_dir
kci_korean_json_filepath = parameters.kci_korean_json_filepath
preprocessed_json_filepath = parameters.preprocessed_json_filepath
whole_units_for_train_txt_filepath = parameters.whole_units_for_train_txt_filepath
model_param_pkl_filepath = parameters.model_param_pkl_filepath
model_json_filepath = parameters.model_json_filepath
model_weights_h5_filepath = parameters.model_weights_h5_filepath
save_epoch_file = parameters.save_epoch_file
model_embedding_vectors_pkl_filepath = parameters.model_embedding_vectors_pkl_filepath

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
rnn_size = parameters.rnn_size
dropout = parameters.dropout
learning_rate = parameters.learning_rate
max_grad_norm = parameters.max_grad_norm
max_epochs = parameters.max_epochs
decay_when = parameters.decay_when
learning_rate_decay = parameters.learning_rate_decay
save_every = parameters.save_every
min_df = parameters.min_df
remove_len = parameters.remove_len

def load_raw_data(filepath):
    with open(filepath,"r") as f:
        d = json.load(f)
    df = pd.read_json(d)
    return df

def raw2sentences(preprocessor, df):
    df = preprocessor.remove_outlier_document(df)
    df = preprocessor.remove_outlier_sentence(df)
    
    # json으로 저장
    with open(preprocessed_json_filepath % len(df), 'w+') as json_file:
        json.dump(df.to_json(orient='records'), json_file)
        print('Created file:', preprocessed_json_filepath % len(df))
    return df

def main():
    df = load_raw_data(kci_korean_json_filepath)
    
    preprocessor = Preprocessor()
    df = raw2sentences(preprocessor, df)
    
    df['flattened_sentences'] = df.apply(lambda x: ' '.join(x['sentences']),axis=1)
    stopwords = preprocessor.stopwords(df['flattened_sentences'], min_df)
    
    print('Extracting nouns..')
    df['nouns'] = df.apply(lambda x: preprocessor.line2words_nouns(x['flattened_sentences'], stopwords, remove_len=remove_len), axis=1)
    
    whole_sentences = preprocessor.flatten_whole_sentences(df, 'nouns')
    print('# of documents = %d' % len(whole_sentences))
    
    # save as .txt
    f = open(whole_units_for_train_txt_filepath, 'w')
    for i in range(len(whole_sentences)):
        data = "%s\n" % whole_sentences[i]
        f.write(data)
    f.close()
    print('Created file:', whole_units_for_train_txt_filepath)
    
    # process text to tensor
    loader = Quantizer(whole_sentences)
    word_vocab_size = min(n_words, len(loader.idx2word))
    char_vocab_size = min(n_chars, len(loader.idx2char))
    max_word_l = loader.max_word_l
    print('Word vocab size: %d, Char vocab size: %d, Max word length (incl. padding): %d' % (word_vocab_size, char_vocab_size, max_word_l))
    
    log_content = '\n=====\n# of stopwords=%d \n%s\n=====\n# of unique words=%d \n# of unique chars=%d \nmaximum length of a word=%d \n=====\n' % (len(stopwords), str(stopwords), word_vocab_size, char_vocab_size, max_word_l)
    write_log(log_dir, 'preprocessing_vocab.log', log_content)

    print('creating an LSTM-CNN with', num_layers, 'layers')
    model = LSTMCNN(char_vocab_size, char_vec_size, feature_maps, kernels, batch_size, seq_length, max_word_l, batch_norm, highway_layers, num_layers, rnn_size, dropout, word_vocab_size, learning_rate, max_grad_norm)
        
    pickle.dump(parameters, open(model_param_pkl_filepath, "wb"))
    model.save(model_json_filepath)
    
    Train, Validation, Test = 0, 1, 2
    model.fit_generator(loader.next_batch(Train), loader.split_sizes[Train], max_epochs, loader.next_batch(Validation), loader.split_sizes[Validation], decay_when, learning_rate_decay, save_every, save_epoch_file)
    model.save_weights(model_weights_h5_filepath, overwrite=True)
    
    # word embedding vectors
#     embedding_tensor = model.layers[-1].weights[0].value()
#     with tf.Session() as sess:
#         init = tf.global_variables_initializer()
#         sess.run(init)
#         embedding = sess.run(embedding_tensor)
#     embedding = np.transpose(embedding)
#     # save as .pkl
#     with open(model_embedding_vectors_pkl_filepath, mode='wb') as f:
#         pickle.dump(embedding, f)
#     print('Created %s' % model_embedding_vectors_pkl_filepath)
    
if __name__ == '__main__':
    main()