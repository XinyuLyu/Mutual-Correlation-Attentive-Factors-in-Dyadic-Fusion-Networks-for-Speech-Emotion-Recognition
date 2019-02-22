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

audio_att_gap = GlobalAveragePooling1D()(audio_att)
dropout_audio = Dropout(0.5)(audio_att_gap)
model_frame = Model(audio_input, dropout_audio)

word_input = Input(shape=(50, 513, 64))
audio_input = TimeDistributed(model_frame)(word_input)
word_att = Attention(4, 16)([audio_input, audio_input, audio_input])
word_att = BatchNormalization()(word_att)
word_att = Attention(4, 16)([word_att, word_att, word_att])

word_att = BatchNormalization()(word_att)
word_att_gap = GlobalAveragePooling1D()(word_att)
dropout_word = Dropout(0.5)(word_att_gap)
audio_prediction = Dense(5, activation='softmax')(dropout_word)
audio_model = Model(inputs=word_input, outputs=audio_prediction)
inter_audio_model = Model(inputs=word_input, outputs=[word_att])
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
audio_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

train_audio_inter = None
test_audio_inter = None
audio_acc = 0
size = 100
epoch = np.linspace(0,size,size+1) #load
#epoch = np.linspace(1,size,size) # no load
loss = []
acc = []

audio_model.load_weights(r'E:\Yue\Code\ACL_entire\audio_model\audio_adam.h5')
inter_audio_model.load_weights(r'E:\Yue\Code\ACL_entire\audio_model\inter_audio_adam.h5')
loss_a, acc_a = audio_model.evaluate_generator(
    data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
    steps=len(test_audio_data) / 4)
loss.append(loss_a)
acc.append(acc_a)
audio_acc = acc_a
train_audio_inter = inter_audio_model.predict_generator(
    data_generator_output(audio_path, train_audio_data, train_label, len(train_audio_data)),
    steps=len(train_audio_data))
test_audio_inter = inter_audio_model.predict_generator(
    data_generator_output(audio_path, test_audio_data, test_label, len(test_audio_data)), steps=len(test_audio_data))

for i in range(size):
    print('audio branch, epoch: ', str(i))
    train_d, train_l = shuffle(train_audio_data, train_label)
    audio_model.fit_generator(data_generator(audio_path, train_d, train_l, len(train_d)),
                              steps_per_epoch=len(train_d) / 4, epochs=1, verbose=1)
    loss_a, acc_a = audio_model.evaluate_generator(
        data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
        steps=len(test_audio_data) / 4)
    loss.append(loss_a)
    acc.append(acc_a)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    if acc_a >= audio_acc:
        audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_model\audio_adam.h5')
        inter_audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_model\inter_audio_ada.h5')
        audio_acc = acc_a
        train_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, train_audio_data, train_label, len(train_audio_data)),
            steps=len(train_audio_data))
        test_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, test_audio_data, test_label, len(test_audio_data)),
            steps=len(test_audio_data))
print(audio_acc)
plt.figure()
plt.plot(epoch, loss, label='loss')
plt.plot(epoch, acc, label ='acc')
plt.xlabel("epoch")
plt.ylabel("loss and acc")
plt.legend()
plt.show()
print('loss:',loss)
print('acc',acc)
