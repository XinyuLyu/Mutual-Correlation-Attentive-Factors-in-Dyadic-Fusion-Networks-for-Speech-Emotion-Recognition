from __future__ import print_function
from self_attention_hybrid import Attention
from DataLoader_4class import get_data, analyze_data, data_generator, data_generator_output  # process_train_data
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling1D, TimeDistributed, Dropout, BatchNormalization
from keras.optimizers import Adam
import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
max_features = 20000
batch_size = 16
epo = 100
filters = 128
flag = 0.60
numclass = 4
audio_path = r'E:\\Yue\\Entire Data\\ACL_2018_entire\\Word_Mat_New_1\\'

# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()


# Audio branch
audio_input = Input(shape=(513, 64))
audio_att1 = Attention(n_head=4, d_k=5)([audio_input, audio_input, audio_input])
audio_att2 = Attention(n_head=4, d_k=5)([audio_att1, audio_att1, audio_att1])
audio_att_gap = GlobalAveragePooling1D()(audio_att2)
model_frame = Model(audio_input, audio_att_gap)

word_input = Input(shape=(50, 513, 64))
word_input1 = TimeDistributed(model_frame)(word_input)
word_att1 = Attention(n_head=4, d_k=10)([word_input1, word_input1, word_input1])
word_att2 = Attention(n_head=4, d_k=10)([word_att1, word_att1, word_att1])
word_att_gap = GlobalAveragePooling1D()(word_att2)#可以试一下不要的话
audio_prediction = Dense(4, activation='softmax')(word_att_gap)
audio_model = Model(inputs=word_input, outputs=audio_prediction)
inter_audio_model = Model(inputs=word_input, outputs=word_att2)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


train_audio_inter = None
test_audio_inter = None
audio_acc = 0
loss =[]
acc = []
size = 100
epoch = np.linspace(1,size,size)
for i in range(size):
    print('audio branch, epoch: ', str(i))
    history = audio_model.fit_generator(data_generator(audio_path, train_audio_data, train_label, len(train_audio_data)),
                                        steps_per_epoch=len(train_audio_data) / 4, epochs=1, verbose=1)
    loss.append(history.history['loss'])
    acc.append(history.history['acc'])
    loss_a, acc_a = audio_model.evaluate_generator(
        data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
        steps=len(test_audio_data) / 4)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    if acc_a >= audio_acc:
        audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_model\audio_model_4_class.h5')
        inter_audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_model\inter_audio_model_4_class.h5')
        audio_acc = acc_a
        train_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, train_audio_data, train_label, len(train_audio_data)),
            steps=len(train_audio_data))
        test_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, test_audio_data, test_label, len(test_audio_data)),
            steps=len(test_audio_data))
print(audio_acc)
'''
audio_model.load_weights(r'E:\Yue\Code\ACL_entire\audio_model\audio_model_4class_best.h5')
result = audio_model.predict([test_audio_data], batch_size=batch_size)
result = np.argmax(result, axis=1)
r_0, r_1, r_2, r_3 = analyze_data(test_label_o, result)
print('final result: ')
print(' audio acc: ', audio_acc)
print("0", r_0)
print("1", r_1)
print("2", r_2)
print("3", r_3)
'''

plt.figure()
plt.plot(epoch, loss, label='loss')
plt.plot(epoch, acc, label='acc')
plt.xlabel("epoch")
plt.ylabel("audio loss and acc")
plt.legend()
plt.show()