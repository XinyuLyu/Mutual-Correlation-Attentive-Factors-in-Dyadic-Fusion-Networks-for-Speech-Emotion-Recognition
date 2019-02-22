from __future__ import print_function
from self_attention_hybrid import Position_Embedding,MultiHeadAttention
from DataLoader_5class import get_data, analyze_data, data_generator, data_generator_output  # process_train_data
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, concatenate, \
    GlobalAveragePooling1D, GlobalMaxPooling1D, TimeDistributed,Lambda
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam
import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from keras import backend
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
# Audio branch
audio_input = Input(shape=(513, 64))
audio_input = Position_Embedding()(audio_input)
audio_att = MultiHeadAttention(n_head=75, d_k=20, d_model=64, dropout=0.25, attn_dropout=0.1, ffn_dropout=0.1, use_norm=True, use_ffn=False)(audio_input, audio_input, audio_input)
audio_att1 = MultiHeadAttention(n_head=75, d_k=20, d_model=64, dropout=0.25, attn_dropout=0.1, ffn_dropout=0.1, use_norm=True, use_ffn=False)(audio_att, audio_att, audio_att)
audio_att2 = MultiHeadAttention(n_head=75, d_k=20, d_model=64, dropout=0.25, attn_dropout=0.1, ffn_dropout=0.1, use_norm=True, use_ffn=False)(audio_att1, audio_att1, audio_att1)
audio_att3 = Lambda(lambda x: backend.sum(x, axis=1))(audio_att1)
audio_att_gap = GlobalMaxPooling1D()(audio_att3)
model_frame = Model(audio_input, audio_att_gap)

word_input = Input(shape=(50, 513, 64))
word_input = Position_Embedding()(word_input)
word_input1 = TimeDistributed(model_frame)(word_input)
word_att = MultiHeadAttention(n_head=50, d_k=5, d_model=64, dropout=0.25, attn_dropout=0.1, ffn_dropout=0.1, use_norm=True, use_ffn=False)(word_input1, word_input1, word_input1)
word_att1 = MultiHeadAttention(n_head=50, d_k=5, d_model=64, dropout=0.25, attn_dropout=0.1, ffn_dropout=0.1, use_norm=True, use_ffn=False)(word_att, word_att, word_att)
word_att2 = MultiHeadAttention(n_head=50, d_k=5, d_model=64, dropout=0.25, attn_dropout=0.1, ffn_dropout=0.1, use_norm=True, use_ffn=False)(word_att1, word_att1, word_att1)
word_att3 = Lambda(lambda x: backend.sum(x, axis=1))(word_att1)
word_att_gap = GlobalMaxPooling1D()(word_att3)
audio_prediction = Dense(5, activation='softmax')(word_att_gap)
audio_model = Model(inputs=word_input, outputs=audio_prediction)
inter_audio_model = Model(inputs=word_input, outputs=word_att1)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_audio_inter = None
test_audio_inter = None
audio_acc = 0
loss =[]
acc = []
size = 100
epoch = np.linspace(1,size,size)
for i in range(size):
    print('audio branch, epoch: ', str(i))
    train_d, train_l = shuffle(train_audio_data, train_label)
    history = audio_model.fit_generator(data_generator(audio_path, train_d, train_l, len(train_d)),
                              steps_per_epoch=len(train_d) / 4, epochs=1, verbose=1)
    loss.append(history.history['loss'])
    acc.append(history.history['acc'])
    loss_a, acc_a = audio_model.evaluate_generator(
        data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
        steps=len(test_audio_data) / 4)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    if acc_a >= audio_acc:
        audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_model\audio_model_5_class.h5')
        inter_audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_model\inter_audio_model_5_class.h5')
        audio_acc = acc_a
        train_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, train_audio_data, train_label, len(train_audio_data)),
            steps=len(train_audio_data))
        test_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, test_audio_data, test_label, len(test_audio_data)),
            steps=len(test_audio_data))
result = audio_model.predict([test_audio_data],batch_size=batch_size)
result = np.argmax(result, axis=1)
r_0, r_1, r_2, r_3, r_4 = analyze_data(test_label_o, result)
print('final result: ')
print(' audio acc: ', audio_acc)
print("0", r_0)
print("1", r_1)
print("2", r_2)
print("3", r_3)
print("4", r_4)

plt.figure()
plt.plot(epo, loss, label='loss')
plt.plot(epo, acc, label= 'acc')
plt.xlabel("epoch")
plt.ylabel("audio loss and acc")
plt.legend()
plt.show()