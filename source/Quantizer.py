from os import path

class Quantizer:
    def __init__(self, tokens, data_dir, batch_size, seq_length, max_word_l, n_words, n_chars):
        self.n_words = n_words
        self.n_chars = n_chars
        train_file = path.join(data_dir, 'train.txt')
        valid_file = path.join(data_dir, 'valid.txt')
        test_file = path.join(data_dir, 'test.txt')
        input_files = [train_file, valid_file, test_file]
        vocab_file = path.join(data_dir, 'vocab.npz')
        tensor_file = path.join(data_dir, 'data')
        char_file = path.join(data_dir, 'data_char')

    # construct a tensor with all the data
    if not (path.exists(vocab_file) or path.exists(tensor_file) or path.exists(char_file)):
        print('one-time setup: preprocessing input train/valid/test files in dir:', data_dir)
        self.text_to_tensor(tokens, input_files, vocab_file, tensor_file, char_file, max_word_l)

    def text_to_tensor(self, tokens, input_files, out_vocabfile, out_tensorfile, out_charfile, max_word_l):
        print('Processing text into tensors...')
        max_word_l_tmp = 0 # max word length of the corpus
        idx2word = [tokens.UNK] # unknown word token
        word2idx = OrderedDict()
        word2idx[tokens.UNK] = 0
        idx2char = [tokens.ZEROPAD, tokens.START, tokens.END, tokens.UNK] # zero-pad, start-of-word, end-of-word tokens
        char2idx = OrderedDict()
        char2idx[tokens.ZEROPAD] = 0
        char2idx[tokens.START] = 1
        char2idx[tokens.END] = 2
        char2idx[tokens.UNK] = 3
        split_counts = []

        # first go through train/valid/test to get max word length
        # if actual max word length is smaller than specified
        # we use that instead. this is inefficient, but only a one-off thing so should be fine
        # also counts the number of tokens
        prog = re.compile('\s+')
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
                charcount.update(word)

            f = codecs.open(input_files[split], 'r', encoding)
            counts = 0
            for line in f:
                line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
                line = line.replace(tokens.START, '')  # start-of-word token is reserved
                line = line.replace(tokens.END, '')  # end-of-word token is reserved
                words = prog.split(line)
                for word in filter(None, words):
                    update(word)
                    max_word_l_tmp = max(max_word_l_tmp, len(word) + 2) # add 2 for start/end chars
                    counts += 1
                if tokens.EOS != '':
                    update(tokens.EOS)
                    counts += 1 # PTB uses \n for <eos>, so need to add one more token at the end
            f.close()
            split_counts.append(counts)

        print('Most frequent words:', len(wordcount))
        for ii, ww in enumerate(wordcount.most_common(self.n_words - 1)):
            word = ww[0]
            word2idx[word] = ii + 1
            idx2word.append(word)
            if ii < 3: print(word)

        print('Most frequent chars:', len(charcount))
        for ii, cc in enumerate(charcount.most_common(self.n_chars - 4)):
            char = cc[0]
            char2idx[char] = ii + 4
            idx2char.append(char)
            if ii < 3: print(char)

        print('Char counts:')
        for ii, cc in enumerate(charcount.most_common()):
            print(ii, cc[0].encode(encoding), cc[1])

        print('After first pass of data, max word length is:', max_word_l_tmp)
        print('Token count: train %d, val %d, test %d' % (split_counts[0], split_counts[1], split_counts[2]))

        # if actual max word length is less than the limit, use that
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
                chars += [char2idx[char] for char in word if char in char2idx]
                chars.append(char2idx[tokens.END]) # end-of-word symbol
                if len(chars) >= max_word_l:
                    chars[max_word_l-1] = char2idx[tokens.END]
                    output_chars[word_num] = chars[:max_word_l]
                else:
                    output_chars[word_num, :len(chars)] = chars
                return word_num + 1

            f = codecs.open(input_files[split], 'r', encoding)
            word_num = 0
            for line in f:
                line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
                line = line.replace(tokens.START, '')  # start-of-word token is reserved
                line = line.replace(tokens.END, '')  # end-of-word token is reserved
                words = prog.split(line)
                for rword in filter(None, words):
                    word_num = append(rword, word_num)
                if tokens.EOS != '':   # PTB does not have <eos> so we add a character for <eos> tokens
                    word_num = append(tokens.EOS, word_num)   # other datasets don't need this
            f.close()
            tensorfile_split = "{}_{}.npy".format(out_tensorfile, split)
            print('saving', tensorfile_split)
            np.save(tensorfile_split, output_tensor)
            charfile_split = "{}_{}.npy".format(out_charfile, split)
            print('saving', charfile_split)
            np.save(charfile_split, output_chars)

        # save output preprocessed files
        print('saving', out_vocabfile)
        np.savez(out_vocabfile, idx2word=idx2word, word2idx=word2idx, idx2char=idx2char, char2idx=char2idx)