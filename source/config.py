from utils import *
import os

# User configuration
kci_korean_document_length_outlier_short = 60
kci_korean_sentence_length_outlier_short = 35
kci_korean_semtemce_length_outlier_long = 8000

# System configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('base_dir:', base_dir)
data_dir = os.path.join(base_dir, 'data')
output_base_dir = os.path.join(base_dir, 'output')
create_dirs([output_base_dir])


# filepath
kci_korean_json_filepath = os.path.join(data_dir, 'kci_korean_sentences_510_191230.json')
preprocessed_json_filepath = os.path.join(output_base_dir, ('%s_preprocessed.json' % now_time_str()))
whole_sentences_txt_filepath = os.path.join(output_base_dir, ('%s_whole_sentences.txt' % now_time_str()))

class Parameters:
    def __init__(self):
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.output_base_dir = output_base_dir
        self.kci_korean_document_length_outlier_short = kci_korean_document_length_outlier_short
        self.kci_korean_sentence_length_outlier_short = kci_korean_sentence_length_outlier_short
        self.kci_korean_semtemce_length_outlier_long = kci_korean_semtemce_length_outlier_long
        self.kci_korean_json_filepath = kci_korean_json_filepath
        self.preprocessed_json_filepath = preprocessed_json_filepath
        self.whole_sentences_txt_filepath = whole_sentences_txt_filepath

    def __str__(self):
        item_strf = ['{} = {}'.format(attribute, value) for attribute, value in self.__dict__.items()]
        strf = 'Parameters(\n  {}\n)'.format('\n  '.join(item_strf))
        return strf


parameters = Parameters()
print(parameters)