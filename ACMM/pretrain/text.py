from __future__ import print_function

from sklearn.utils import shuffle

from self_attention_hybrid import Position_Embedding, Attention
from Document_level_analysis.DataLoader_text import get_data  # process_train_data
from keras.models import Model
from keras.layers import Dense, Input, Embedding, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras import backend
import numpy as np

import matplotlib.pyplot as plt

max_features = 20000
batch_size = 32
epo = 100
filters = 128
flag = 0.60
numclass = 4

def compute_acc(predict, label):
    acc = 0
    # print(predict.shape, label.shape)
    for l in range(len(label)):
        if np.argmax(predict[l]) == np.argmax(label[l]):
            acc += 1
    return acc / len(label)
def reshape(x):
    backend.permute_dimensions(x, (0, 2, 1))
    return x


# loading data
print('Loading data...')
train_text_data, train_label, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()

# Text Branch (adam)
text_input = Input(shape=(50,))
em_text = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)
em_text = Position_Embedding()(em_text)
text_att = Attention(n_head=4, d_k=16)([em_text, em_text, em_text])
text_att1 = Attention(n_head=4, d_k=16)([text_att, text_att, text_att])
text_att_gap = GlobalMaxPooling1D()(text_att1)
'''
audio_inputs = Lambda(reshape)(text_att1)
text_att_gap = Bidirectional(LSTM(128, return_sequences=False, recurrent_dropout=0.25))(text_att1)
'''
text_prediction = Dense(10)(text_att_gap)
text_model = Model(inputs=text_input, outputs=text_prediction)
inter_text_model = Model(inputs=text_input, outputs=text_att_gap)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
text_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])

text_mae = 0
text_loss = 1000
text_acc = 0

loss_test = []
mae_test = []
loss_train = []


size = 1000
epoch = np.linspace(1, size, size)  # load

for i in range(size):
    print('text branch, epoch: ', str(i))
    train_d, train_l = shuffle(train_text_data, train_label)
    history = text_model.fit(train_d, train_l, batch_size=batch_size, epochs=1, verbose=1)
    loss_train.append(history.history['loss'])
    loss_t, mae_t = text_model.evaluate(test_text_data, test_label, batch_size=batch_size, verbose=0)
    loss_test.append(loss_t)
    mae_test.append(mae_t)
    print('epoch: ', str(i))
    print('loss_t', loss_t, ' ', 'mae_t', mae_t)
    if loss_t < text_loss:
        text_mae = mae_t
        text_loss = loss_t
        inter_text_model.save_weights(r'E:\Yue\Code\ACL_entire\text_model\\inter_text_4_class.h5')
        text_model.save_weights(r'E:\Yue\Code\ACL_entire\text_model\\text_4_class.h5')
        rest_test = text_model.predict(test_text_data, batch_size=batch_size)
        acc = compute_acc(rest_test, test_label)
        text_acc = acc

print('mae: ', text_mae,'loss',text_loss,'acc',text_acc)
inter_text_model.load_weights(r'E:\Yue\Code\ACL_entire\text_model\\inter_text_4_class.h5')
train_text_inter = inter_text_model.predict(train_text_data, batch_size=1)
np.save(r'E:\Yue\Code\ACL_entire\train.npy',train_text_inter)
test_text_inter = inter_text_model.predict(test_text_data, batch_size=1)
np.save(r'E:\Yue\Code\ACL_entire\test.npy',test_text_inter)

plt.figure()
plt.plot(epoch, loss_train, label='loss_train')
plt.plot(epoch, loss_test, label='loss_test')
plt.plot(epoch, mae_test, label='mae_test')
plt.xlabel("epoch")
plt.ylabel("loss and mae")
plt.legend()
plt.show()


