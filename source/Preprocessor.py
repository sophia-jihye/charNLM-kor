from config import parameters
from utils import *
import os
import re
import pandas as pd

letter_pattern = re.compile('[^ㄱ-ㅣ가-힣a-zA-Z]+')
doublespace_pattern = re.compile('\s+')

class Preprocessor:
    def __init__(self):
        self.kci_korean_document_length_outlier_short = parameters.kci_korean_document_length_outlier_short
        self.kci_korean_sentence_length_outlier_short = parameters.kci_korean_sentence_length_outlier_short
        self.kci_korean_semtemce_length_outlier_long = parameters.kci_korean_semtemce_length_outlier_long
        
    def __str__(self):
        return os.path.abspath(__file__)
    
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
    
    def flatten_whole_sentences(self, df):
        whole_sentences = list()
        for index, row in df.iterrows():
            whole_sentences.extend(row['sentences'])
        return whole_sentences