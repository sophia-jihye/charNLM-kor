import numpy as np
from os import path
import gc
import re
from collections import Counter, OrderedDict, namedtuple
from config import parameters
from utils import *
import hangul

class Quantizer:
    def __init__(self, whole_sentences):
        self.log_dir = parameters.log_dir
        self.prog = re.compile('\s+')
        self.tokens = self.tokens()
        self.batch_size = parameters.batch_size
        self.seq_length = parameters.seq_length
        self.n_words = parameters.n_words
        self.n_chars = parameters.n_chars
        input_objects = [whole_sentences, whole_sentences, whole_sentences[:100]]
        vocab_filepath = parameters.vocab_filepath
        tensor_file = parameters.tensor_file
        char_file = parameters.char_file
        initial_max_word_l = parameters.max_word_l
        
        self.text_to_tensor(self.tokens, input_objects, vocab_filepath, tensor_file, char_file, initial_max_word_l)
        
        self.idx2word, self.word2idx, self.idx2char, self.char2idx = self.vocab_unpack(vocab_filepath)
        self.vocab_size = len(self.idx2word)
        self.word_vocab_size = len(self.idx2word)
        print('Word vocab size: %d, Char vocab size: %d' % (len(self.idx2word), len(self.idx2char)))
        
        print('loading data files...')
        all_data, all_data_char = self.load_tensor_data(tensor_file, char_file)
        self.max_word_l = all_data_char[0].shape[1]   # create word-char mappings
        
        print('reshaping tensors...')
        self.batch_idx = [0,0,0]
        self.data_sizes, self.split_sizes, self.all_batches = self.reshape_tensors(all_data, all_data_char, self.batch_size, self.seq_length)

        print('data load done. Number of batches in train: %d, val: %d, test: %d'
              % (self.split_sizes[0], self.split_sizes[1], self.split_sizes[2]))
        gc.collect()
            
    def tokens(self):
        Tokens = namedtuple('Tokens', ['EOS', 'UNK', 'START', 'END', 'ZEROPAD'])
        tokens = Tokens(
                EOS='`',
                UNK='|',    # unk word token
                START='{',  # start-of-word token
                END='}',    # end-of-word token
                ZEROPAD=' ' # zero-pad token
            )
        return tokens
    
    def vocab_unpack(self, vocab_filepath):
        vocab = np.load(vocab_filepath)
        return vocab['idx2word'].item(), vocab['word2idx'].item(), vocab['idx2char'].item(), vocab['char2idx'].item()
    
    def load_tensor_data(self, tensor_file, char_file):
        all_data = []
        all_data_char = []
        for split in range(3):
            all_data.append(np.load("{}_{}.npy".format(tensor_file, split)))  # train, valid, test tensors
            all_data_char.append(np.load("{}_{}.npy".format(char_file, split)))  # train, valid, test character indices
        return all_data, all_data_char
    
    def reshape_tensors(self, all_data, all_data_char, batch_size, seq_length):
        data_sizes = []
        split_sizes = []
        all_batches = []
        for split, data in enumerate(all_data):
            data_len = data.shape[0]
            data_sizes.append(data_len)
            if split < 2 and data_len % (batch_size * seq_length) != 0:
                data = data[:batch_size * seq_length * (data_len // (batch_size * seq_length))]
            ydata = data.copy()
            ydata[0:-1] = data[1:]
            ydata[-1] = data[0]
            data_char = all_data_char[split][:len(data)]
            if split < 2:
                rdata = data.reshape((batch_size, -1))
                rydata = ydata.reshape((batch_size, -1))
                rdata_char = data_char.reshape((batch_size, -1, self.max_word_l))
            else: # for test we repeat dimensions to batch size (easier but inefficient evaluation)
                nseq = (data_len + (seq_length - 1)) // seq_length
                rdata = data.copy()
                rdata.resize((1, nseq*seq_length))
                rdata = np.tile(rdata, (batch_size, 1))
                rydata = ydata.copy()
                rydata.resize((1, nseq*seq_length))
                rydata = np.tile(rydata, (batch_size, 1))
                rdata_char = data_char.copy()
                rdata_char.resize((1, nseq*seq_length, rdata_char.shape[1]))
                rdata_char = np.tile(rdata_char, (batch_size, 1, 1))
            # split in batches
            x_batches = np.split(rdata, rdata.shape[1] // seq_length, axis=1)
            y_batches = np.split(rydata, rydata.shape[1] // seq_length, axis=1)
            x_char_batches = np.split(rdata_char, rdata_char.shape[1] // seq_length, axis=1)
            nbatches = len(x_batches)
            split_sizes.append(nbatches)
            assert len(x_batches) == len(y_batches)
            assert len(x_batches) == len(x_char_batches)
            all_batches.append((x_batches, y_batches, x_char_batches))
        return data_sizes, split_sizes, all_batches
    
    def word2jamo(self, word):
        l = [hangul.split_jamo(char) for char in word]
        sum(l, [])
        jamo_list = list()
        for jamo in sum(l,[]):
            jamo_list.extend(jamo)
        return jamo_list
    
    def line2words_blank(self, line):
        words = self.prog.split(line)
        return words

    def text_to_tensor(self, tokens, input_objects, out_vocabfile, out_tensorfile, out_charfile, max_word_l):
        print('Processing text into tensors...')
        max_word_l_tmp = 0 # max word length of the corpus
        
        word2idx = {tokens.UNK:0}        
        char2idx = {tokens.UNK:0, tokens.START:1, tokens.END:2, tokens.ZEROPAD:3}
        
        split_counts = []
        
        wordcount = Counter()
        charcount = Counter()
        for	split in range(3): # split = 0 (train), 1 (val), or 2 (test)

            def update(word):
                if word[0] == tokens.UNK:
                    if len(word) > 1: # unk token with character info available
                        word = word[2:]
                else:
                    wordcount.update([word])
                word = word.replace(tokens.UNK, '')
                charcount.update(self.word2jamo(word))

            counts = 0
            for line in input_objects[split]:
                line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
                line = line.replace(tokens.START, '')  # start-of-word token is reserved
                line = line.replace(tokens.END, '')  # end-of-word token is reserved
                words = self.line2words_blank(line)
                for word in filter(None, words):
                    update(word)
                    max_word_l_tmp = max(max_word_l_tmp, len(word) + 2) # add 2 for start/end chars
                    counts += 1
                if tokens.EOS != '':
                    update(tokens.EOS)
                    counts += 1 # PTB uses \n for <eos>, so need to add one more token at the end
            split_counts.append(counts)

        print('# of unique words: %d' % len(wordcount))
        write_log(self.log_dir, 'num_of_unique_tokens.log', len(wordcount))
        for ii, ww in enumerate(wordcount.most_common(self.n_words - 1)):
            word = ww[0]
            word2idx[word] = ii + 1

        print('# of unique characters: %d' % len(charcount))
        for ii, cc in enumerate(charcount.most_common(self.n_chars - 4)):
            char = cc[0]
            char2idx[char] = ii + 4

        print('After first pass of data, max word length is:', max_word_l_tmp)
        print('# of tokens (not unique): %d' % split_counts[0] )
        write_log(self.log_dir, 'num_of_tokens.log', split_counts[0])
        
        # REAL MAXIMUM WORD LENGTH
        max_word_l = min(max_word_l_tmp, max_word_l)

        for split in range(3):  # split = 0 (train), 1 (val), or 2 (test)
            # Preallocate the tensors we will need.
            # Watch out the second one needs a lot of RAM.
            output_tensor = np.empty(split_counts[split], dtype='int32')
            output_chars = np.zeros((split_counts[split], max_word_l), dtype='int32')

            def append(word, word_num):
                chars = [char2idx[tokens.START]] # start-of-word symbol
                if word[0] == tokens.UNK and len(word) > 1: # unk token with character info available
                    word = word[2:]
                    output_tensor[word_num] = word2idx[tokens.UNK]
                else:
                    output_tensor[word_num] = word2idx[word] if word in word2idx else word2idx[tokens.UNK]
                chars += [char2idx[char] for char in self.word2jamo(word) if char in char2idx]
                chars.append(char2idx[tokens.END]) # end-of-word symbol
                if len(chars) >= max_word_l:
                    chars[max_word_l-1] = char2idx[tokens.END]
                    output_chars[word_num] = chars[:max_word_l]
                else:
                    output_chars[word_num, :len(chars)] = chars
                return word_num + 1

            word_num = 0
            for line in input_objects[split]:
                line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
                line = line.replace(tokens.START, '')  # start-of-word token is reserved
                line = line.replace(tokens.END, '')  # end-of-word token is reserved
                words = self.line2words_blank(line)
                for rword in filter(None, words):
                    word_num = append(rword, word_num)
                if tokens.EOS != '':   # PTB does not have <eos> so we add a character for <eos> tokens
                    word_num = append(tokens.EOS, word_num)   # other datasets don't need this
            tensorfile_split = "{}_{}.npy".format(out_tensorfile, split)
            print('saving', tensorfile_split)
            np.save(tensorfile_split, output_tensor)
            charfile_split = "{}_{}.npy".format(out_charfile, split)
            print('saving', charfile_split)
            np.save(charfile_split, output_chars)

        # save output preprocessed files
        idx2word = dict([(value, key) for (key, value) in word2idx.items()])
        idx2char = dict([(value, key) for (key, value) in char2idx.items()])
        print('saving', out_vocabfile)
        np.savez(out_vocabfile, idx2word=idx2word, word2idx=word2idx, idx2char=idx2char, char2idx=char2idx)
        
    def next_batch(self, split_idx):
        while True:
            # split_idx is integer: 0 = train, 1 = val, 2 = test
            self.batch_idx[split_idx] += 1
            if self.batch_idx[split_idx] >= self.split_sizes[split_idx]:
                self.batch_idx[split_idx] = 0 # cycle around to beginning

            # pull out the correct next batch
            idx = self.batch_idx[split_idx]
            word = self.all_batches[split_idx][0][idx]
            sparse_ydata = self.all_batches[split_idx][1][idx]
            chars = self.all_batches[split_idx][2][idx]
            # expand dims for sparse_cross_entropy optimization
            ydata = np.expand_dims(sparse_ydata, axis=2)

            yield ({'word':word, 'chars':chars}, ydata)