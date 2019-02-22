from __future__ import print_function
from self_attention_hybrid import Attention, Position_Embedding
from load_final_data_2 import get_data, data_generator, data_generator_output  # process_train_data
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, concatenate, \
    GlobalAveragePooling1D, Conv1D, GlobalMaxPooling1D, TimeDistributed
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam
import numpy as np
from keras import backend
from sklearn.utils import shuffle
from sklearn import preprocessing

max_features = 20000
batch_size = 16
epo = 100
filters = 128
flag = 0.60
numclass = 5
audio_path = r'E:\\Yue\\Entire Data\\ACL_2018_entire\\Word_Mat_New_1\\'

# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()

print('train_audio shape:', len(train_audio_data))
print('train_text shape:', train_text_data.shape)
print('test_audio shape:', len(test_audio_data))
print('test_text shape:', test_text_data.shape)
print('train_label shape:', train_label.shape)
print('test_label shape:', test_label.shape)


def weight_expand(x):
    return backend.expand_dims(x)


def weight_dot(inputs):
    x = inputs[0]
    y = inputs[1]
    return x * y


def weight_average(inputs):
    x = inputs[0]
    y = inputs[1]
    return (x + y) / 2


def data_normal(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x

# Audio branch
audio_input = Input(shape=(513, 64))
print('audio_input shape: ', audio_input.shape)
audio_att = Attention(4, 16)([audio_input, audio_input, audio_input])
audio_att = BatchNormalization()(audio_att)
audio_att = Attention(4, 16)([audio_att, audio_att, audio_att])
audio_att = BatchNormalization()(audio_att)
audio_att_gap = GlobalAveragePooling1D()(audio_att)
print('audio_att shape: ', audio_att.shape)
dropout_audio = Dropout(0.5)(audio_att_gap)
model_frame = Model(audio_input, dropout_audio)

word_input = Input(shape=(98, 513, 64))
print('word_input shape: ', word_input.shape)
audio_input = TimeDistributed(model_frame)(word_input)
word_att = Attention(4, 16)([audio_input, audio_input, audio_input])
word_att = BatchNormalization()(word_att)
word_att = Attention(4, 16)([word_att, word_att, word_att])
word_att = BatchNormalization()(word_att)
word_att_gap = GlobalAveragePooling1D()(word_att)
print('word_att',word_att.shape)
dropout_word = Dropout(0.5)(word_att_gap)
d_1_a = Dense(128)(dropout_word)
batch_nol1_a = BatchNormalization()(d_1_a)
activation1_a = Activation('relu')(batch_nol1_a)
dropout_activation1_a = Dropout(0.5)(activation1_a)

d_2_a = Dense(64)(dropout_activation1_a)
batch_nol2_a = BatchNormalization()(d_2_a)
activation2_a = Activation('relu')(batch_nol2_a)
dropout_activation2_a = Dropout(0.5)(activation2_a)

d_3_a = Dense(32)(dropout_activation2_a)
batch_nol3_a = BatchNormalization()(d_3_a)
activation3_a = Activation('relu')(batch_nol3_a)
dropout_activation3_a = Dropout(0.5)(activation3_a)

d_4_a = Dense(16)(dropout_activation3_a)
batch_nol4_a = BatchNormalization()(d_4_a)
activation4_a = Activation('relu')(batch_nol4_a)
dropout_activation4_a = Dropout(0.5)(activation4_a)

d_5_a = Dense(8)(dropout_activation4_a)
batch_nol5_a = BatchNormalization()(d_5_a)
activation5_a = Activation('relu')(batch_nol5_a)
dropout_activation5_a = Dropout(0.5)(activation5_a)

d_6_a = Dense(1)(dropout_activation5_a)
batch_nol6_a = BatchNormalization()(d_6_a)
activation6_a = Activation('relu')(batch_nol6_a)
dropout_activation6_a = Dropout(0.5)(activation6_a)
print('dropout_activation6_a', dropout_activation6_a.shape)

audio_prediction = Dense(5, activation='softmax')(dropout_activation6_a)
audio_model = Model(inputs=word_input, outputs=audio_prediction)
inter_audio_model = Model(inputs=word_input, outputs=[word_att])
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

'''
# Audio branch
word_input = Input(shape=(513, 64))
#word_input = Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.25, name='LSTM_audio_1'))(word_input)
word_att = Attention(4, 16)([word_input, word_input, word_input])
word_att = BatchNormalization()(word_att)
word_att = Attention(4, 16)([word_att, word_att, word_att])
word_att = BatchNormalization()(word_att)
word_att = GlobalAveragePooling1D()(word_att)
dropout_word = Dropout(0.5)(word_att)
model_frame = Model(word_input, dropout_word)

audio_input = Input(shape=(98, 513, 64))
audio_input = TimeDistributed(model_frame)(audio_input)
#audio_input = Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.25, name='LSTM_audio_2'))(audio_input)
audio_att = Attention(4, 16)([audio_input, audio_input, audio_input])
audio_att = BatchNormalization()(audio_att)
audio_att1 = Attention(4, 16)([audio_att, audio_att, audio_att])
audio_att1 = BatchNormalization()(audio_att1)
audio_att1 = GlobalAveragePooling1D()(audio_att1)
#audio_att2 = Attention(4, 16)([audio_att1, audio_att1, audio_att1])
#audio_att2 = BatchNormalization()(audio_att2)
dropout_audio_att2 = Dropout(0.5)(audio_att1)

audio_prediction = Dense(5, activation='softmax')(dropout_audio_att2)
audio_model = Model(inputs=audio_input, outputs=audio_prediction)
inter_audio_model = Model(inputs=audio_input, outputs=[audio_att1])#######
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
'''
# Text Branch
text_input = Input(shape=(98,))
em_text = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)
em_text = Position_Embedding()(em_text)
text_att = Attention(10, 20)([em_text, em_text, em_text])
text_att = BatchNormalization()(text_att)

text_att1 = Attention(10, 20)([text_att, text_att, text_att])
text_att1 = BatchNormalization()(text_att1)

text_att2 = Attention(10, 20)([text_att1, text_att1, text_att1])
text_att2 = BatchNormalization()(text_att2)
print('text_att2',text_att2.shape)
text_att_gap = GlobalAveragePooling1D()(text_att2)
dropout_text_gap = Dropout(0.5)(text_att_gap)
print('text_att_gap',text_att_gap.shape)
'''
d_1_t = Dense(128)(dropout_text_gap)
print('d_1_t',d_1_t.shape)
batch_nol1_t = BatchNormalization()(d_1_t)
print('batch_nol1_t',batch_nol1_t.shape)
activation1_t = Activation('relu')(batch_nol1_t)
print('activation1_t',activation1_t.shape)
dropout_activation1_t = Dropout(0.5)(activation1_t)
print('dropout_activation1_t', dropout_activation1_t.shape)

d_2_t = Dense(64)(dropout_activation1_t)
batch_nol2_t = BatchNormalization()(d_2_t)
activation2_t = Activation('relu')(batch_nol2_t)
dropout_activation2_t = Dropout(0.5)(activation2_t)

d_3_t = Dense(32)(dropout_activation2_t)
batch_nol3_t = BatchNormalization()(d_3_t)
activation3_t = Activation('relu')(batch_nol3_t)
dropout_activation3_t = Dropout(0.5)(activation3_t)

d_4_t = Dense(16)(dropout_activation3_t)
batch_nol4_t = BatchNormalization()(d_4_t)
activation4_t = Activation('relu')(batch_nol4_t)
dropout_activation4_t = Dropout(0.5)(activation4_t)

d_5_t = Dense(8)(dropout_activation4_t)
batch_nol5_t = BatchNormalization()(d_5_t)
activation5_t = Activation('relu')(batch_nol5_t)
dropout_activation5_t = Dropout(0.5)(activation5_t)

d_6_t = Dense(1)(dropout_activation5_t)
batch_nol6_t = BatchNormalization()(d_6_t)
activation6_t = Activation('relu')(batch_nol6_t)
dropout_activation6_t = Dropout(0.5)(activation6_t)
print('dropout_activation6_t',dropout_activation6_t.shape)
'''
text_prediction = Dense(5, activation='softmax')(text_att_gap)
print('text_prediction',text_prediction.shape)
text_model = Model(inputs=text_input, outputs=text_prediction)
inter_text_model = Model(inputs=text_input, outputs=text_att2)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
text_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Fusion Model
audio_f_input = Input(shape=(98,64))  # 50，    #98,200      #98,
text_f_input = Input(shape=(98,200))  # ，64 #98,200      #98,513,64
merge = concatenate([text_f_input, audio_f_input], name='merge')
merge = Dropout(0.5)(merge)
print('merge shape: ', merge.shape)    # (?,98,264)

merge_weight = Attention(10,20)([merge,merge,merge])
merge = BatchNormalization()(merge_weight)
#merge_att_gap = GlobalAveragePooling1D()(merge)
#dropout_merge_gap = Dropout(0.5)(merge_att_gap)   #(?,264)
print('merge_att_gap',merge.shape)

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


d_1 = Dense(256)(final_merge)
print('d_1',d_1.shape)
batch_nol1 = BatchNormalization()(d_1)
print('batch_nol1',batch_nol1.shape)
activation1 = Activation('relu')(batch_nol1)
print('activation1',activation1.shape)
d_drop1 = Dropout(0.6)(activation1)
print('d_drop1',d_drop1.shape)

d_2 = Dense(128)(d_drop1)
print('d_2',d_2.shape)
batch_nol2 = BatchNormalization()(d_2)
print('batch_nol2',batch_nol2.shape)
activation2 = Activation('relu')(batch_nol2)
print('activation2',activation2.shape)
d_drop2 = Dropout(0.6)(activation2)
print('d_drop2',d_drop2.shape)

f_prediction = Dense(5, activation='softmax')(d_drop2)
print('f_prediction',f_prediction.shape)
final_model = Model(inputs=[text_f_input, audio_f_input], outputs=f_prediction)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
final_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
final_inter_model = Model(inputs=[text_f_input, audio_f_input], outputs=merge_weight)

text_acc = 0
train_text_inter = None
test_text_inter = None
for i in range(50):
    print('text branch, epoch: ', str(i))
    text_model.fit(train_text_data, train_label, batch_size=batch_size, epochs=1, verbose=1)
    loss_t, acc_t = text_model.evaluate(test_text_data, test_label, batch_size=batch_size, verbose=0)
    print('epoch: ', str(i))
    print('loss_t', loss_t, ' ', 'acc_t', acc_t)
    if acc_t >= text_acc:
        text_acc = acc_t
        train_text_inter = inter_text_model.predict(train_text_data, batch_size=batch_size)
        test_text_inter = inter_text_model.predict(test_text_data, batch_size=batch_size)
        text_model.save_weights(r'E:\Yue\Code\ACL_entire\text_model\\text_model.h5')
        inter_text_model.save_weights(r'E:\Yue\Code\ACL_entire\text_model\\inter_text_model.h5')
train_audio_inter = None
test_audio_inter = None
audio_acc = 0
for i in range(0):
    print('audio branch, epoch: ', str(i))
    train_d, train_l = shuffle(train_audio_data, train_label)
    audio_model.fit_generator(data_generator(audio_path, train_d, train_l, len(train_d)),
                              steps_per_epoch=len(train_d) / 4, epochs=1, verbose=1)
    loss_a, acc_a = audio_model.evaluate_generator(
        data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
        steps=len(test_audio_data) / 4)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    if acc_a >= audio_acc:
        audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_model\audio_model.h5')
        inter_audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_model\inter_audio_model.h5')
        audio_acc = acc_a
        train_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, train_audio_data, train_label,len(train_audio_data)),steps=len(train_audio_data))
        test_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, test_audio_data, test_label,len(test_audio_data)),steps=len(test_audio_data))
'''inter_audio_model.load_weights(r'E:\Yue\Code\ACL_entire\audio_model\inter_audio_model.h5')
train_audio_inter = inter_audio_model.predict_generator(data_generator_output(audio_path, train_audio_data, train_label,
                                                                              len(train_audio_data)),
                                                        steps=len(train_audio_data))
test_audio_inter = inter_audio_model.predict_generator(data_generator_output(audio_path, test_audio_data, test_label,
                                                                             len(test_audio_data)),
                                                       steps=len(test_audio_data))
                                                       '''
final_acc = 0
result = None
for i in range(0):
    print('fusion branch, epoch: ', str(i))
    final_model.fit([train_text_inter, train_audio_inter], train_label, batch_size=batch_size, epochs=1)
    loss_f, acc_f = final_model.evaluate([test_text_inter, test_audio_inter], test_label, batch_size=batch_size,
                                         verbose=0)
    print('epoch: ', str(i))
    print('loss_f', loss_f, ' ', 'acc_f', acc_f)
    if acc_f >= final_acc:
        final_model.save_weights(r'E:\Yue\Code\ACL_entire\final_model\final_model.h5')
        final_inter_model.save_weights(r'E:\Yue\Code\ACL_entire\final_model\final_inter_model.h5')
        final_acc = acc_f
        result = final_model.predict([test_text_inter, test_audio_inter], batch_size=batch_size)
        test_fusion_weight = final_inter_model.predict([test_text_inter, test_audio_inter], batch_size=batch_size)
        result = np.argmax(result, axis=1)

#r_0, r_1, r_2, r_3, r_4 = analyze_data(test_label_o, result)
print('final result: ')
print('text acc',text_acc)
'''
print('text acc: ', text_acc, ' audio acc: ', audio_acc, ' final acc: ', final_acc)
print("0", r_0)
print("1", r_1)
print("2", r_2)
print("3", r_3)
print("4", r_4)
'''