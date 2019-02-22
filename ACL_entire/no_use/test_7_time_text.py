from __future__ import print_function
from self_attention_hybrid import Attention
from DataLoader_5class import get_data, data_generator, data_generator_output,analyze_data  # process_train_data
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, concatenate, \
    GlobalAveragePooling1D, Conv1D, GlobalMaxPooling1D, TimeDistributed
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils import shuffle
import datetime
max_features = 20000
batch_size = 16
epo = 100
filters = 128
flag = 0.60
numclass = 5
audio_path = r'E:\\Yue\\Entire Data\\ACL_2018_entire\\Word_Mat_New_1\\'

# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic, token_data = get_data()



print('train_audio shape:', len(train_audio_data))
print('train_text shape:', train_text_data.shape)
print('test_audio shape:', len(test_audio_data))
print('test_text shape:', test_text_data.shape)
print('train_label shape:', train_label.shape)
print('test_label shape:', test_label.shape)

# Text Branch
text_input = Input(shape=(50,))
em_text = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)
# em_text = Position_Embedding()(em_text)

text_att = Attention(10, 20)([em_text, em_text, em_text])
text_att = BatchNormalization()(text_att)

text_att1 = Attention(10, 20)([text_att, text_att, text_att])
text_att1 = BatchNormalization()(text_att1)

text_att2 = Attention(10, 20)([text_att1, text_att1, text_att1])
text_att2 = BatchNormalization()(text_att2)

text_att_gap = GlobalMaxPooling1D()(text_att2)
# text_att_gap = GlobalAveragePooling1D()(text_att2)
# dropout_text_gap = Dropout(0.5)(text_att_gap)

text_prediction = Dense(5, activation='softmax')(text_att_gap)
text_model = Model(inputs=text_input, outputs=text_prediction)
inter_text_model = Model(inputs=text_input, outputs=text_att2)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#text_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
text_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

text_acc = 0
train_text_inter = None
test_text_inter = None

for i in range(100):
    start_time=None
    end_time=None
    print('text branch, epoch: ', str(i))
    start_time = datetime.datetime.now()
    text_model.fit(train_text_data, train_label, batch_size=batch_size, epochs=1, verbose=1)
    end_time = datetime.datetime.now()
    loss_t, acc_t = text_model.evaluate(test_text_data, test_label, batch_size=batch_size, verbose=0)
    print('epoch: ', str(i))
    print('loss_t', loss_t, ' ', 'acc_t', acc_t)
    print(end_time.second-start_time.second)
    if acc_t >= text_acc:
        text_acc = acc_t
        train_text_inter = inter_text_model.predict(train_text_data, batch_size=batch_size)
        test_text_inter = inter_text_model.predict(test_text_data, batch_size=batch_size)
        text_model.save_weights(r'E:\Yue\Code\ACL_entire\text_model\\sgd2.h5')
        inter_text_model.save_weights(r'E:\Yue\Code\ACL_entire\text_model\\inter_sgd2.h5')
print(text_acc)
