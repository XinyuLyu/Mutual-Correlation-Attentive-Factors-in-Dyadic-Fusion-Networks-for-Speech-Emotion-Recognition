from __future__ import print_function
from Document_level_analysis.DataLoader_audio import get_data
from keras.models import Model
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import Activation
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras import backend
import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

batch_size = 32
dropout = 0.25
save_path = r'E:\Yue\Code\ACL_entire\Document_level_analysis\\'


def compute_acc(predict, label):
    acc = 0
    # print(predict.shape, label.shape)
    for l in range(len(label)):
        if np.argmax(predict[l]) == np.argmax(label[l]):
            acc += 1
    return acc / len(label)


def save_list(path, data):
    file = open(path, 'w')
    file.write(str(data))
    file.close()


def reshape(x):
    backend.permute_dimensions(x, (0, 2, 1))
    return x


def expand_dimensions(x):
    return backend.expand_dims(x)


def transpose(x):
    return backend.transpose(x)


# loading data
print('Loading data...')
train_audio_data, train_label, test_audio_data, test_label = get_data()

# sentence-level feature extraction
audio_input = Input(shape=(6553, ))

audio_inputs = Lambda(expand_dimensions)(audio_input)
cnn_1 = Conv1D(32, 128, padding='valid')(audio_inputs)
cnn_1 = BatchNormalization()(cnn_1)
cnn_1 = Activation('relu')(cnn_1)
cnn_1 = MaxPooling1D(pool_size=3)(cnn_1)

cnn_2 = Conv1D(64, 64, padding='valid')(cnn_1)
cnn_2 = BatchNormalization()(cnn_2)
cnn_2 = Activation('relu')(cnn_2)
cnn_2 = MaxPooling1D(pool_size=3)(cnn_2)

cnn_3 = Conv1D(128, 32, padding='valid')(cnn_2)
cnn_3 = BatchNormalization()(cnn_3)
cnn_3 = Activation('relu')(cnn_3)
cnn_3 = MaxPooling1D(pool_size=3)(cnn_3)

cnn_4 = Conv1D(256, 16, padding='valid')(cnn_3)
cnn_4 = BatchNormalization()(cnn_4)
cnn_4 = Activation('relu')(cnn_4)
cnn_4 = MaxPooling1D(pool_size=3)(cnn_4)

cnn_5 = Conv1D(512, 8, padding='valid')(cnn_4)
cnn_5 = BatchNormalization()(cnn_5)
cnn_5 = Activation('relu')(cnn_5)
cnn_5 = MaxPooling1D(pool_size=3)(cnn_5)

audio_att_gap = Flatten()(cnn_5)

audio_att_gap = Dense(2048)(audio_att_gap)
audio_att_gap = BatchNormalization()(audio_att_gap)
audio_att_gap = Activation('relu')(audio_att_gap)

audio_att_rep = Dense(200)(audio_att_gap)
audio_att_rep = BatchNormalization()(audio_att_rep)
audio_att_rep = Activation('relu')(audio_att_rep)

# cnn_5 = Lambda(reshape)(cnn_5)
# audio_att = Attention(n_head=4, d_k=16)([cnn_5, cnn_5, cnn_5])
# audio_att = Attention(n_head=4, d_k=16)([audio_att, audio_att, audio_att])
# audio_att_rep = GlobalMaxPooling1D()(audio_att)

audio_prediction = Dense(10)(audio_att_rep)
audio_model = Model(inputs=audio_input, outputs=audio_prediction)
inter_audio_model = Model(inputs=audio_input, outputs=audio_att_rep)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
audio_model.summary()


audio_acc = 0
audio_mae = 0
audio_loss = 1000
loss_train = []
mae_train = []
loss_test = []
mae_test = []
size = 1000
epoch = np.linspace(1, size, size)

# Sentence-level
for i in range(size):
    print('audio branch, epoch: ', str(i))
    epo_data, epo_label = shuffle(train_audio_data, train_label)
    history = audio_model.fit(epo_data, epo_label, batch_size=batch_size, epochs=1, verbose=1)
    loss_train.append(history.history['loss'])
    loss_a, mae_a = audio_model.evaluate(test_audio_data, test_label, batch_size=batch_size, verbose=0)
    mae_test.append(mae_a)
    loss_test.append(loss_a)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', mae_a)
    if loss_a <= audio_loss:
        audio_mae = mae_a
        audio_loss = loss_a
        inter_audio_model.save_weights(save_path + 'inter_audio_regression.h5')
        audio_model.save_weights(save_path + 'audio_regression.h5')
        output_test = audio_model.predict(test_audio_data, batch_size=batch_size)
        acc = compute_acc(output_test, test_label)
        audio_acc = acc
    print('mae: ', audio_mae, 'loss', audio_loss, 'acc', audio_acc)


print('Best: ', 'mae: ', audio_mae, 'loss', audio_loss, 'acc', audio_acc)
inter_audio_model.load_weights(save_path + 'inter_audio_regression.h5')
train_audio_inter = inter_audio_model.predict(train_audio_data, batch_size=1)
np.save(save_path + 'audio_train.npy', train_audio_inter)
test_audio_inter = inter_audio_model.predict(test_audio_data, batch_size=1)
np.save(save_path + 'audio_test.npy', test_audio_inter)

plt.figure()
plt.plot(epoch, loss_train, label='loss_train')
plt.plot(epoch, loss_test, label='loss_test')
plt.plot(epoch, mae_test, label='mae_test')
plt.xlabel("epoch")
plt.ylabel("loss and mae")
plt.legend()
plt.show()
