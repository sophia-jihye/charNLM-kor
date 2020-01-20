from config import parameters
from utils import *
import os
import re
import pandas as pd
import numpy as np
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer

letter_pattern = re.compile('[^ㄱ-ㅣ가-힣a-zA-Z]+')
doublespace_pattern = re.compile('\s+')

class Preprocessor:
    def __init__(self):
        self.log_dir = parameters.log_dir
        self.kci_korean_document_length_outlier_short = parameters.kci_korean_document_length_outlier_short
        self.kci_korean_sentence_length_outlier_short = parameters.kci_korean_sentence_length_outlier_short
        self.kci_korean_semtemce_length_outlier_long = parameters.kci_korean_semtemce_length_outlier_long
        self.okt = Okt()
        
    def __str__(self):
        return os.path.abspath(__file__)
    
    def line2words_nouns(self, line, stopwords=None, remove_len=False):
        words = [word for (word, pos) in self.okt.pos(line) if pos in ['Alpha', 'Noun']]
        if stopwords is not None:
            words = [word for word in words if word not in stopwords]
        if remove_len:
            words = [word for word in words if len(word) != 1]
        return words
    
    def remove_outlier_document(self, df):
        df['length_of_doc'] = df.apply(lambda x: len(x['sentences']),axis=1)
        short_length_doc_index = df[df['length_of_doc'] <= self.kci_korean_document_length_outlier_short].index
        print('# of short length of document outliers:', len(short_length_doc_index))
        df = df.drop(short_length_doc_index)
        print('final # of documents: ', len(df))
        return df
    
    def remove_outlier_sentence(self, df):
        for index, row in df.iterrows():
            i = 0
            while i < len(row['sentences']):
                len_of_sentence = len(row['sentences'][i])
                if len_of_sentence <= self.kci_korean_sentence_length_outlier_short:
                    del row['sentences'][i]
                    i -= 1
                elif len_of_sentence >= self.kci_korean_semtemce_length_outlier_long:
                    del row['sentences'][i]
                    i -= 1
                i += 1
        return df
    
    def stopwords(self, corpus, min_df):
        print('Creating stopwords..')
        vectorizer = CountVectorizer(min_df=min_df, max_df=1.0, tokenizer=lambda x:self.line2words_nouns(x))
        X = vectorizer.fit_transform(corpus)
        stopwords = list(vectorizer.vocabulary_.keys())
        
        print('Creating stopwords..')
        vectorizer = CountVectorizer(min_df=0.0, max_df=1.0, tokenizer=lambda x:self.line2words_nouns(x))
        X = vectorizer.fit_transform(corpus)
        freq = np.sum(X.toarray(), axis=0)
        for word_idx in range(len(vectorizer.vocabulary_.keys())):
            if freq[word_idx] <= 1:
                stopwords.append(list(vectorizer.vocabulary_.keys())[word_idx])
        return stopwords
    
    def flatten_whole_sentences(self, df, key_column):
        whole_sentences = list()
        for index, row in df.iterrows():
            whole_sentences.append(' '.join(row[key_column]))
        return whole_sentences