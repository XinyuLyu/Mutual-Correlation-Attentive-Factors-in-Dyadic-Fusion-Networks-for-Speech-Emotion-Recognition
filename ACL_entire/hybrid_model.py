from __future__ import print_function
from self_attention_hybrid import Position_Embedding, Attention, FusionAttention
from DataLoader_hybrid_4class import get_data, data_generator, data_generator_output, analyze_data  # process_train_data
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, concatenate
from keras.layers import GlobalAveragePooling1D, Conv1D, GlobalMaxPooling1D, TimeDistributed, BatchNormalization
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Lambda
from keras import backend as K
def save_list(path,data):
    file = open(path, 'w')
    file.write(str(data))
    file.close()
max_features = 20000
batch_size = 16
epo = 100
filters = 128
flag = 0.60
numclass = 4
audio_path = r'E:\\Yue\\Entire Data\\ACL_2018_entire\\Word_Mat_New_1\\'
text_path = r'E:/Yue/Entire Data/ACL_2018_entire/text_output_new.txt'
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

audio_input = Input(shape=(513, 64))
audio_att1 = Attention(n_head=4, d_k=16)([audio_input, audio_input, audio_input])
audio_att2 = Attention(n_head=4, d_k=16)([audio_att1, audio_att1, audio_att1])
audio_att_gap = GlobalMaxPooling1D()(audio_att2)
model_frame = Model(audio_input, audio_att_gap)

word_input = Input(shape=(50, 513, 64))
text_input = Input(shape=(50,))
word_input1 = TimeDistributed(model_frame)(word_input)
text_input1 = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)
text_input1 = Position_Embedding()(text_input1)
fusion_att_a, fusion_att_t = FusionAttention(n_head=4, d_k=16)([text_input1, text_input1, text_input1, word_input1, word_input1, word_input1])
fusion_att_a1, fusion_att_t1 = FusionAttention(n_head=4, d_k=16)([fusion_att_a, fusion_att_a, fusion_att_a, fusion_att_t, fusion_att_t, fusion_att_t])
fusion_att_a2, fusion_att_t2 = FusionAttention(n_head=4, d_k=16)([fusion_att_a1, fusion_att_a1, fusion_att_a1, fusion_att_t1, fusion_att_t1, fusion_att_t1])
concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))
fusion_att_concat = concat([fusion_att_a2, fusion_att_t2])
fusion_att = GlobalMaxPooling1D()(fusion_att_concat)
fusion_prediction = Dense(4, activation='softmax')(fusion_att)
fusion_model = Model(inputs=[word_input, text_input], outputs=fusion_prediction)
adam = Adam(lr=0.0007, beta_1=0.9, beta_2=0.98, epsilon=1e-09)#0.001,0.9,0.999,10-8
fusion_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

fusion_acc = 0
train_fusion_inter = None
test_fusion_inter = None

loss = []
acc = []
size = 100
epoch = np.linspace(1, size, size)  # load

print('start adam')
for i in range(size):
    print('branch, epoch: ', str(i))
    history = fusion_model.fit_generator(
        data_generator(audio_path, train_audio_data, train_text_data, train_label, len(train_audio_data)),
        steps_per_epoch=len(train_audio_data) / 4, epochs=1, verbose=1)
    loss.append(history.history['loss'])
    acc.append(history.history['acc'])
    loss_f, acc_f = fusion_model.evaluate_generator(data_generator(audio_path, test_audio_data,test_text_data, test_label, len(test_audio_data)),steps=len(test_audio_data) / 4)
    print('epoch: ', str(i))
    print('loss_f', loss_f, ' ', 'acc_f', acc_f)
    if acc_f >= fusion_acc:
        fusion_acc = acc_f
        fusion_model.save_weights(r'E:\Yue\Code\ACL_entire\fusion_model\\fusion_4_class.h5')
        model_frame.save_weights(r'E:\Yue\Code\ACL_entire\fusion_model\\frame_4_class.h5')
        result = fusion_model.predict_generator(
            data_generator_output(audio_path, test_audio_data, test_text_data, test_label, len(test_audio_data)),
            steps=len(test_audio_data))
        result = np.argmax(result, axis=1)
print('fusion ends: ', fusion_acc)

r_0, r_1, r_2, r_3 = analyze_data(test_label_o, result)
print('final result: ')
print('text acc: ', fusion_acc)
print("0", r_0)
print("1", r_1)
print("2", r_2)
print("3", r_3)
plt.figure()
plt.plot(epoch, loss, label='loss')
plt.plot(epoch, acc, label='acc')
plt.xlabel("epoch")
plt.ylabel("loss and acc")
plt.legend()
plt.show()

save_list(r'E:\Yue\Code\ACL_entire\hybrid\acc.txt',acc)
save_list(r'E:\Yue\Code\ACL_entire\hybrid\loss',loss)
