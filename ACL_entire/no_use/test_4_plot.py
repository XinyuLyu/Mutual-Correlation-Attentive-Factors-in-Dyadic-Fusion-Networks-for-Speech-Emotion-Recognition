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
import matplotlib.pyplot as plt
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

# Audio branch
audio_input = Input(shape=(513, 64))
audio_att = Attention(4, 16)([audio_input, audio_input, audio_input])
audio_att = BatchNormalization()(audio_att)
audio_att = Attention(4, 16)([audio_att, audio_att, audio_att])
audio_att = BatchNormalization()(audio_att)

#audio_att_gap = GlobalAveragePooling1D()(audio_att)
audio_att_gap = GlobalMaxPooling1D()(audio_att)
dropout_audio = Dropout(0.5)(audio_att_gap)
model_frame = Model(audio_input, dropout_audio)

word_input = Input(shape=(50, 513, 64))
audio_input = TimeDistributed(model_frame)(word_input)
word_att = Attention(4, 16)([audio_input, audio_input, audio_input])
word_att = BatchNormalization()(word_att)
word_att = Attention(4, 16)([word_att, word_att, word_att])
word_att = BatchNormalization()(word_att)

#word_att_gap = GlobalAveragePooling1D()(word_att)
word_att_gap = GlobalMaxPooling1D()(word_att)
dropout_word = Dropout(0.5)(word_att_gap)
audio_prediction = Dense(5, activation='softmax')(dropout_word)
audio_model = Model(inputs=word_input, outputs=audio_prediction)
inter_audio_model = Model(inputs=word_input, outputs=[word_att])
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
audio_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
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

# Fusion Model
audio_f_input = Input(shape=(50, 64))  # 50，    #98,200      #98,
text_f_input = Input(shape=(50, 200))  # ，64 #98,200      #98,513,64
merge = concatenate([text_f_input, audio_f_input], name='merge')
merge = Dropout(0.5)(merge)

merge_weight1 = Attention(10, 20)([merge, merge, merge])
merge_weight1 = BatchNormalization()(merge_weight1)

merge_weight2 = Attention(10, 20)([merge_weight1, merge_weight1, merge_weight1])
merge_weight2 = BatchNormalization()(merge_weight2)

merge_weight3 = Attention(10, 20)([merge_weight2, merge_weight2, merge_weight2])
merge = BatchNormalization()(merge_weight3)  # (?,98,264)
merge_gmp = GlobalMaxPooling1D()(merge)
'''
cnn_1 = Conv1D(filters, 2, padding='valid', strides=1)(merge)
batchnol1 = BatchNormalization()(cnn_1)
activation1 = Activation('relu')(batchnol1)
maxpool_1 = GlobalMaxPooling1D()(activation1)
dropout_1 = Dropout(0.7)(maxpool_1)

cnn_2 = Conv1D(filters, 3, padding='valid', strides=1)(merge)
batchnol2 = BatchNormalization()(cnn_2)
activation2 = Activation('relu')(batchnol2)
maxpool_2 = GlobalMaxPooling1D()(activation2)
dropout_2 = Dropout(0.7)(maxpool_2)

cnn_3 = Conv1D(filters, 4, padding='valid', strides=1)(merge)
batchnol3 = BatchNormalization()(cnn_3)
activation3 = Activation('relu')(batchnol3)
maxpool_3 = GlobalMaxPooling1D()(activation3)
dropout_3 = Dropout(0.7)(maxpool_3)

cnn_4 = Conv1D(filters, 5, padding='valid', strides=1)(merge)
batchnol4 = BatchNormalization()(cnn_4)
activation4 = Activation('relu')(batchnol4)
maxpool_4 = GlobalMaxPooling1D()(activation4)
dropout_4 = Dropout(0.7)(maxpool_4)
final_merge = concatenate([dropout_1, dropout_2, dropout_3, dropout_4], name='final_merge')
'''
d_1 = Dense(256)(merge_gmp)
batch_nol1 = BatchNormalization()(d_1)
activation1 = Activation('relu')(batch_nol1)
d_drop1 = Dropout(0.6)(activation1)

d_2 = Dense(128)(d_drop1)
batch_nol2 = BatchNormalization()(d_2)
activation2 = Activation('relu')(batch_nol2)
d_drop2 = Dropout(0.6)(activation2)

f_prediction = Dense(5, activation='softmax')(d_drop2)
print('f_prediction', f_prediction.shape)
final_model = Model(inputs=[text_f_input, audio_f_input], outputs=f_prediction)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
final_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
final_inter_model = Model(inputs=[text_f_input, audio_f_input], outputs=merge_weight3)

text_acc = 0
train_text_inter = None
test_text_inter = None
size = 50
epoch = np.linspace(0,size,size+1) #load
#epoch = np.linspace(1,size,size) # no load
loss = []
acc = []

text_model.load_weights(r'E:\Yue\Code\ACL_entire\text_model\\text_model_1_27_adam.h5')
inter_text_model.load_weights(r'E:\Yue\Code\ACL_entire\text_model\\inter_text_model_1_27_adam.h5')
loss_t, acc_t = text_model.evaluate(test_text_data, test_label, batch_size=batch_size, verbose=0)
loss.append(loss_t)
acc.append(acc_t)
text_acc = acc_t
train_text_inter = inter_text_model.predict(train_text_data, batch_size=batch_size)
test_text_inter = inter_text_model.predict(test_text_data, batch_size=batch_size)
print('loss', loss)
print('acc', acc)


for i in range(size):
    print('text branch, epoch: ', str(i))
    data_, label_ = shuffle(train_text_data, train_label)
    text_model.fit(data_, label_, batch_size=batch_size, epochs=1, verbose=1)
    data_test, label_test = shuffle(test_text_data,test_label)
    loss_t, acc_t = text_model.evaluate(data_test, label_test, batch_size=batch_size, verbose=0)
    print('epoch: ', str(i))
    print('loss_t', loss_t, ' ', 'acc_t', acc_t)
    loss.append(loss_t)
    acc.append(acc_t)
    if acc_t >= text_acc:
        text_acc = acc_t
        train_text_inter = inter_text_model.predict(data_, batch_size=batch_size)
        test_text_inter = inter_text_model.predict(data_test, batch_size=batch_size)
        #text_model.save_weights(r'E:\Yue\Code\ACL_entire\text_model\\text_model_1_27_adam.h5')
        #inter_text_model.save_weights(r'E:\Yue\Code\ACL_entire\text_model\\inter_text_model_1_27_adam.h5')
plt.figure()
plt.plot(epoch, loss, label='loss')
plt.plot(epoch, acc, label ='acc')
plt.xlabel("epoch")
plt.ylabel("loss and acc")
plt.legend()
plt.show()
print('loss:',loss)
print('acc',acc)


