from __future__ import print_function
from self_attention_hybrid import Position_Embedding, Attention, FusionAttention
from DataLoader_hybrid import get_data, data_generator, data_generator_output, analyze_data,get_data1
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, concatenate, Activation
from keras.layers import GlobalAveragePooling1D, Conv1D, GlobalMaxPooling1D, TimeDistributed, BatchNormalization
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Lambda
from keras import backend as K
from keras import backend
from data import save_list
max_features = 20000
batch_size = 8
epo = 100
filters = 128
flag = 0.60
numclass = 4
audio_path = r'E:\\Yue\\Entire Data\\ACL_2018_entire\\Word_Mat_New_1\\'
text_path = r'E:/Yue/Entire Data/ACL_2018_entire/text_output_new.txt'


# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()

#frame model
audio_input = Input(shape=(513, 64))
audio_input_tran = K.permute_dimensions(audio_input, (0, 2, 1))
audio_att1 = Attention(n_head=4, d_k=10)([audio_input_tran, audio_input_tran, audio_input_tran])
audio_att1 = Dropout(0.1)(audio_att1)
audio_att2 = Attention(n_head=4, d_k=10)([audio_att1, audio_att1, audio_att1])
audio_att2 = Dropout(0.1)(audio_att2)
audio_att_gap = Lambda(lambda x: backend.sum(x, axis=1))(audio_att2)
audio_att_gap = Dense(64)(audio_att_gap)
audio_att_gap = Activation('tanh')(audio_att_gap)
model_frame = Model(audio_input, audio_att_gap)

#audio frame
word_input = Input(shape=(50, 513, 64))
text_input = Input(shape=(50,))
word_input1 = TimeDistributed(model_frame)(word_input)
text_input1 = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)
text_input1 = Position_Embedding()(text_input1)
word_input1 = Position_Embedding()(word_input1)
fusion_att_a, fusion_att_t = FusionAttention(n_head=4, d_k=10, name='a_1')([word_input1, word_input1, word_input1, text_input1, text_input1, text_input1])
#fusion_att_a = BatchNormalization()(fusion_att_a)
#fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_a = Dropout(0.1)(fusion_att_a)
fusion_att_t = Dropout(0.25)(fusion_att_t)
fusion_att_a, fusion_att_t = FusionAttention(n_head=4, d_k=10, name='a_2')([fusion_att_a, fusion_att_a, fusion_att_a, fusion_att_t, fusion_att_t, fusion_att_t])
#fusion_att_a = BatchNormalization()(fusion_att_a)
#fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_a = Dropout(0.1)(fusion_att_a)
fusion_att_t = Dropout(0.25)(fusion_att_t)
fusion_att_a, fusion_att_t = FusionAttention(n_head=4, d_k=10, name='a_3')([fusion_att_a, fusion_att_a, fusion_att_a, fusion_att_t, fusion_att_t, fusion_att_t])#
#fusion_att_a = BatchNormalization()(fusion_att_a)
#fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_a = Dropout(0.1)(fusion_att_a)
fusion_att_t = Dropout(0.25)(fusion_att_t)
fusion_att_a, fusion_att_t = FusionAttention(n_head=4, d_k=10, name='a_4')([fusion_att_a, fusion_att_a, fusion_att_a, fusion_att_t, fusion_att_t, fusion_att_t])
#fusion_att_a = BatchNormalization()(fusion_att_a)
#fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_a = Dropout(0.1)(fusion_att_a)
fusion_att_t = Dropout(0.25)(fusion_att_t)
concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))
fusion_att_concat = concat([fusion_att_a, fusion_att_t])
fusion_att_concat = Lambda(lambda x: backend.sum(x, axis=1))(fusion_att_concat)
#fusion_att_concat = Dense(20)(fusion_att_concat)
#fusion_att_concat_1 = Activation('tanh')(fusion_att_concat)
#fusion_att_concat_1 = GlobalMaxPooling1D()(fusion_att_concat)
fusion_prediction = Dense(4, activation='softmax')(fusion_att_concat)
fusion_model = Model(inputs=[word_input, text_input], outputs=fusion_prediction)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  #0.001,0.9,0.999,10-8

# model_frame.load_weights(r'E:\Yue\Code\ACL_entire\hybrid\\frame_4_class.h5')
# fusion_model.load_weights(r'E:\Yue\Code\ACL_entire\hybrid\\fusion_4_class.h5')

fusion_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fusion_model.summary()

# fusion_loss, fusion_acc = fusion_model.evaluate_generator(data_generator(audio_path, test_audio_data,test_text_data, test_label, len(test_audio_data)),steps=len(test_audio_data) / 4)
# result = fusion_model.predict_generator(data_generator_output(audio_path, test_audio_data, test_text_data, test_label, len(test_audio_data)),steps=len(test_audio_data))
# result = np.argmax(result, axis=1)


# train
fusion_acc = 0
fusion_loss = 0
train_fusion_inter = None
test_fusion_inter = None
loss_train = []
acc_train = []
loss_test = []
acc_test = []
size = 100
epoch = np.linspace(1, size, size)
for i in range(size):
    print('branch, epoch: ', str(i))
    history = fusion_model.fit_generator(
        data_generator(audio_path, train_audio_data, train_text_data, train_label, len(train_audio_data)),
        steps_per_epoch=len(train_audio_data) / batch_size, epochs=1, verbose=1)
    loss_train.append(history.history['loss'])
    acc_train.append(history.history['acc'])
    loss_f, acc_f = fusion_model.evaluate_generator(data_generator(audio_path, test_audio_data,
                                                                   test_text_data, test_label, len(test_audio_data)),
                                                    steps=len(test_audio_data) / batch_size)
    loss_test.append(loss_f)
    acc_test.append(acc_f)
    print('epoch: ', str(i))
    print('loss_f', loss_f, ' ', 'acc_f', acc_f)
    if acc_f >= fusion_acc:
        fusion_acc = acc_f
        fusion_loss = loss_f
        fusion_model.save_weights(r'E:\Yue\Code\ACL_entire\hybrid\\fusion_4_class.h5')
        model_frame.save_weights(r'E:\Yue\Code\ACL_entire\hybrid\\frame_4_class.h5')
        result = fusion_model.predict_generator(
            data_generator_output(audio_path, test_audio_data, test_text_data, test_label, len(test_audio_data)),
            steps=len(test_audio_data))
        result = np.argmax(result, axis=1)


#confusion matrix
r_0, r_1, r_2, r_3 = analyze_data(test_label_o, result)
print('final result: ')
print('fusion acc: ', fusion_acc, 'fusion loss', fusion_loss)
print("0", r_0)
print("1", r_1)
print("2", r_2)
print("3", r_3)


#plot and save plot data

plt.figure()
plt.plot(epoch, loss_train, label='loss_train')
plt.plot(epoch, acc_train, label='acc_train')
plt.plot(epoch, loss_test, label='loss_test')
plt.plot(epoch, acc_test, label='acc_test')
plt.xlabel("epoch")
plt.ylabel("loss and acc for train and test")
plt.legend()
plt.show()

save_list(r'E:\Yue\Code\ACL_entire\hybrid\acc_train.txt', acc_train)
save_list(r'E:\Yue\Code\ACL_entire\hybrid\loss_train.txt', loss_train)
save_list(r'E:\Yue\Code\ACL_entire\hybrid\acc_test.txt', acc_test)
save_list(r'E:\Yue\Code\ACL_entire\hybrid\loss_test.txt', loss_test)
