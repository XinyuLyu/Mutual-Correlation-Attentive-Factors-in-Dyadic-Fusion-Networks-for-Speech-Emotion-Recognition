from __future__ import print_function
from DataLoader_5class import get_data
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix,dic, tokenpre = get_data()
#print(tokenpre)
from bert_serving.client import BertClient
print(len(tokenpre))
print("Start:\n")
bc=BertClient(ip='localhost',port=5555)
print("Connecting:\n")
a=bc.encode(tokenpre)
print(a,"\n",a.shape,"\n")
import numpy as np
np.save('bert_text_pretrained_feather_Cased_768_word_embedding_max_sep_len=98.npy',a)
#import numpy as np
#bert_con3=np.load("bert_text_pretrained_feather_word_embedding_con3.npy")
#bert_5=np.load("bert_text_pretrained_feather_word_embedding_5.npy")
#bert_final=np.concatenate((bert_con3,bert_5),axis=0)
#np.save("bert_text_pretrained_feather_word_embedding_con4.npy",bert_final)
#print(bert_final.shape)