from __future__ import print_function
from self_attention_hybrid import Position_Embedding,Attention
from DataLoader_4class import get_data, analyze_data, data_generator, data_generator_output  # process_train_data
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, concatenate, \
    GlobalAveragePooling1D, GlobalMaxPooling1D, TimeDistributed
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt

max_features = 20000
batch_size = 16
epo = 100
filters = 128
flag = 0.60
numclass = 4
audio_path = r'E:\\Yue\\Entire Data\\ACL_2018_entire\\Word_Mat_New_1\\'

def save_list(path,data):
    file = open(path, 'w')
    file.write(str(data))
    file.close()
# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()
save_list(r'E:\Yue\Code\ACL_entire\validation\train_audio_data.txt',train_audio_data)
np.save(r'E:\Yue\Code\ACL_entire\validation\train_text_data.npy', train_text_data)
np.save(r'E:\Yue\Code\ACL_entire\validation\train_label.npy', train_label)
save_list(r'E:\Yue\Code\ACL_entire\validation\test_audio_data.txt',test_audio_data)
np.save(r'E:\Yue\Code\ACL_entire\validation\test_text_data.npy', test_text_data)
np.save(r'E:\Yue\Code\ACL_entire\validation\test_label.npy', test_label)
np.save(r'E:\Yue\Code\ACL_entire\validation\test_label_o.npy', test_label_o)

# Audio branch
audio_input = Input(shape=(513, 64))
audio_att = Attention(4, 16)([audio_input, audio_input, audio_input])
audio_att1 = Attention(4, 16)([audio_att, audio_att, audio_att])
audio_att_gap = GlobalAveragePooling1D()(audio_att1)
model_frame = Model(audio_input, audio_att_gap)

word_input = Input(shape=(50, 513, 64))
word_input1 = TimeDistributed(model_frame)(word_input)
word_att = Attention(4, 16)([word_input1, word_input1, word_input1])
word_att1 = Attention(4, 16)([word_att, word_att, word_att])
word_att_gap = GlobalAveragePooling1D()(word_att1)
audio_prediction = Dense(4, activation='softmax')(word_att_gap)
audio_model = Model(inputs=word_input, outputs=audio_prediction)
inter_audio_model = Model(inputs=word_input, outputs=word_att1)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Text Branch(adam)
text_input = Input(shape=(50,))
em_text = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)
em_text = Position_Embedding()(em_text)
text_att = Attention(4, 16)([em_text, em_text, em_text])
text_att1 = Attention(4, 16)([text_att, text_att, text_att])
text_att_gap = GlobalMaxPooling1D()(text_att1)
text_prediction = Dense(4, activation='softmax')(text_att_gap)
text_model = Model(inputs=text_input, outputs=text_prediction)
inter_text_model = Model(inputs=text_input, outputs=text_att1)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
text_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fusion Model
audio_f_input = Input(shape=(50, 64))  # 50，    #98,200      #98,
text_f_input = Input(shape=(50, 200))  # ，64 #98,200      #98,513,64
merge = concatenate([text_f_input, audio_f_input], name='merge')
merge = Dropout(0.5)(merge)

merge_gmp = GlobalMaxPooling1D()(merge)
d_1 = Dense(256)(merge_gmp)
batch_nol1 = BatchNormalization()(d_1)
activation1 = Activation('relu')(batch_nol1)
d_drop1 = Dropout(0.6)(activation1)

d_2 = Dense(128)(d_drop1)
batch_nol2 = BatchNormalization()(d_2)
activation2 = Activation('relu')(batch_nol2)
d_drop2 = Dropout(0.6)(activation2)

f_prediction = Dense(4, activation='softmax')(d_drop2)
print('f_prediction', f_prediction.shape)
final_model = Model(inputs=[text_f_input, audio_f_input], outputs=f_prediction)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
final_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

size_t = 100
loss_text = []
acc_text = []
epoch_text = np.linspace(1,size_t,size_t)
text_acc = 0
train_text_inter = None
test_text_inter = None
for i in range(size_t):
    print('text branch, epoch: ', str(i))
    history_t = text_model.fit(train_text_data, train_label, batch_size=batch_size, epochs=1, verbose=1)
    loss_text.append(history_t.history['loss'])
    acc_text.append(history_t.history['acc'])
    loss_t, acc_t = text_model.evaluate(test_text_data, test_label, batch_size=batch_size, verbose=0)
    print('epoch: ', str(i))
    print('loss_t', loss_t, ' ', 'acc_t', acc_t)
    if acc_t >= text_acc:
        text_acc = acc_t
        train_text_inter = inter_text_model.predict(train_text_data, batch_size=batch_size)
        test_text_inter = inter_text_model.predict(test_text_data, batch_size=batch_size)
        text_model.save_weights(r'E:\Yue\Code\ACL_entire\validation\\text_model_4_class.h5')
        inter_text_model.save_weights(r'E:\Yue\Code\ACL_entire\validation\\inter_text_model_4_class.h5')

size_a = 50
loss_audio = []
acc_audio = []
epoch_audio = np.linspace(1,size_a,size_a)
train_audio_inter = None
test_audio_inter = None
audio_acc = 0
for i in range(size_a):
    print('audio branch, epoch: ', str(i))
    history_a=audio_model.fit_generator(data_generator(audio_path, train_audio_data, train_label, len(train_audio_data)),
                              steps_per_epoch=len(train_audio_data) / 4, epochs=1, verbose=1)
    loss_audio.append(history_a.history['loss'])
    acc_audio.append(history_a.history['acc'])
    loss_a, acc_a = audio_model.evaluate_generator(
        data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
        steps=len(test_audio_data) / 4)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    if acc_a >= audio_acc:
        audio_model.save_weights(r'E:\Yue\Code\ACL_entire\validation\audio_model_4_class.h5')
        inter_audio_model.save_weights(r'E:\Yue\Code\ACL_entire\validation\inter_audio_model_4_class.h5')
        model_frame.save_weights(r'E:\Yue\Code\ACL_entire\validation\frame_model_4_class.h5')
        audio_acc = acc_a
        train_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, train_audio_data, train_label, len(train_audio_data)),
            steps=len(train_audio_data))
        test_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, test_audio_data, test_label, len(test_audio_data)),
            steps=len(test_audio_data))
size_f = 100
loss_fusion = []
acc_fusion = []
epoch_fusion = np.linspace(1,size_f,size_f)
final_acc = 0
result = None
for i in range(size_f):
    print('fusion branch, epoch: ', str(i))
    history_f = final_model.fit([train_text_inter, train_audio_inter], train_label, batch_size=batch_size, epochs=1)
    loss_fusion.append(history_f.history['loss'])
    acc_fusion.append(history_f.history['acc'])
    loss_f, acc_f = final_model.evaluate([test_text_inter, test_audio_inter], test_label, batch_size=batch_size,
                                         verbose=0)
    print('epoch: ', str(i))
    print('loss_f', loss_f, ' ', 'acc_f', acc_f)
    if acc_f >= final_acc:
        final_model.save_weights(r'E:\Yue\Code\ACL_entire\fusion_model\fusion_model_4_class.h5')
        final_acc = acc_f
        result = final_model.predict([test_text_inter, test_audio_inter], batch_size=batch_size)
        result = np.argmax(result, axis=1)

r_0, r_1, r_2, r_3 = analyze_data(test_label_o, result)
print('final result: ')
print('text acc: ', text_acc, ' audio acc: ', audio_acc, ' final acc: ', final_acc)
print("0", r_0)
print("1", r_1)
print("2", r_2)
print("3", r_3)

save_list(r'E:\Yue\Code\ACL_entire\validation\acc_audio.txt',acc_audio)
save_list(r'E:\Yue\Code\ACL_entire\validation\acc_text.txt',acc_text)
save_list(r'E:\Yue\Code\ACL_entire\validation\acc_fusion.txt',acc_fusion)
save_list(r'E:\Yue\Code\ACL_entire\validation\loss_text.txt',loss_text)
save_list(r'E:\Yue\Code\ACL_entire\validation\loss_fusion.txt',loss_fusion)
save_list(r'E:\Yue\Code\ACL_entire\validation\loss_audio.txt',loss_audio)

plt.subplot(1,3,1)
plt.title('audio')
plt.plot(epoch_audio, loss_audio, label='loss')
plt.plot(epoch_audio, acc_audio, label='acc')
plt.xlabel("epoch")
plt.ylabel("audio loss and acc")
plt.legend()
plt.subplot(1,3,2)
plt.title('text')
plt.plot(epoch_text, loss_text, label='loss')
plt.plot(epoch_text, acc_text, label='acc')
plt.xlabel("epoch")
plt.ylabel("text loss and acc")
plt.legend()
plt.subplot(1,3,3)
plt.title('fusion')
plt.plot(epoch_fusion, loss_fusion, label='loss')
plt.plot(epoch_fusion, acc_fusion, label='acc')
plt.xlabel("epoch")
plt.ylabel("fusion loss and acc")
plt.legend()
plt.show()
