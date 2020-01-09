from Preprocessor import Preprocessor
from config import parameters
from utils import *
import pandas as pd
import json

kci_korean_json_filepath = parameters.kci_korean_json_filepath
preprocessed_json_filepath = parameters.preprocessed_json_filepath
whole_sentences_txt_filepath = parameters.whole_sentences_txt_filepath

def load_data(filepath):
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
    df = load_data(kci_korean_json_filepath)
    
    preprocessor = Preprocessor()
    df = raw2preprocessed(preprocessor, df)
    whole_sentences = for_train(preprocessor, df)
    
if __name__ == '__main__':
    main()