#change the the input with token, get the feather output and change the input embedding matrix
import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint

'''
if len(sys.argv) != 4:
    print('python load_model.py CONFIG_PATH CHECKPOINT_PATH DICT_PATH')
    print('CONFIG_PATH:     $UNZIPPED_MODEL_PATH/bert_config.json')
    print('CHECKPOINT_PATH: $UNZIPPED_MODEL_PATH/bert_model.ckpt')
    print('DICT_PATH:       $UNZIPPED_MODEL_PATH/vocab.txt')
    sys.exit(-1)

config_path, checkpoint_path, dict_path = tuple(sys.argv[1:])
'''

config_path='E:/Yue/Entire Data/BERT/uncased_L-24_H-1024_A-16/uncased_L-24_H-1024_A-16/bert_config.json'
checkpoint_path='E:/Yue/Entire Data/BERT/uncased_L-24_H-1024_A-16/uncased_L-24_H-1024_A-16/bert_model.ckpt'
dict_path='E:/Yue/Entire Data/BERT/uncased_L-24_H-1024_A-16/uncased_L-24_H-1024_A-16/vocab.txt'
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
#model.summary(line_length=120)

tokens = ['[CLS]', 'i', 'am', 'a', 'student', '[SEP]']

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

token_input = np.asarray([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
seg_input = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens))])

print(token_input[0][:len(tokens)])

predicts = model.predict([token_input, seg_input])[0]
for i, token in enumerate(tokens):
    print("shape:",predicts[i].shape,"\n")
    #print(token, predicts[i].tolist()[:5])
