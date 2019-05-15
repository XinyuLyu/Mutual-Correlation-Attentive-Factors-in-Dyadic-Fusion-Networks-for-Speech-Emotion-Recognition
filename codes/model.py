from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras import backend as K
from keras import regularizers as rl
from keras import Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from Document_level_analysis.session.codes.attention import FusionAttention
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import numpy as np
from Document_level_analysis.session.codes.get_FLOPs import get_flops

data_path = r'.\data\\'
save_path = r'.\model\\'
batch_size = 64
n_head = 10
d_k = 20
activation = 'tanh'
dense_size = 64
dropout = 0.25

acc = 0
mae = 100
loss = 1000
loss_train = []
mae_train = []
loss_test = []
mae_test = []
size = 50
epoch = np.linspace(1, size, size)


def get_data(path):
    print('loading data...')
    train_a = np.load(path + 'train_audio_ori.npy')
    test_a = np.load(path + 'test_audio_ori.npy')
    train_t = np.load(path + 'train_text_ori.npy')
    test_t = np.load(path + 'test_text_ori.npy')
    dev_a = np.load(path + 'dev_audio_ori.npy')
    dev_t = np.load(path + 'dev_text_ori.npy')
    dev_l = np.load(path + 'dev_label.npy')
    train_l = np.load(path + 'train_label.npy')
    test_l = np.load(path + 'test_label.npy')
    print('finish loading data...')

    return train_a, train_t, test_a, test_t, train_l, test_l, dev_a, dev_t, dev_l


def expand_dimensions(x):
    return K.expand_dims(x)


def remove_dimensions(x):
    return K.squeeze(x, axis=1)

# show confusion matrix of predicted result
def analyze_data(label, predict):
    r_0 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_1 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_2 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_3 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_4 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_5 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_6 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_7 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_8 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_9 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    i = 0
    labels = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    while i < len(label):  # 4
        num_l = compute_same_label(label[i])
        num_p = compute_same_label(predict[i])
        if num_l == 0 and num_p == 0:
            if np.argmax(label[i]) == 0:
                labels[str(0)] += 1
                r_0[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i])  == 1:
                labels[str(1)] += 1
                r_1[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i])  == 2:
                labels[str(2)] += 1
                r_2[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i])  == 3:
                labels[str(3)] += 1
                r_3[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i])  == 4:
                labels[str(4)] += 1
                r_4[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i])  == 5:
                labels[str(5)] += 1
                r_5[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i])  == 6:
                labels[str(6)] += 1
                r_6[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i])  == 7:
                labels[str(7)] += 1
                r_7[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i])  == 8:
                labels[str(8)] += 1
                r_8[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i])  == 9:
                labels[str(9)] += 1
                r_9[str(np.argmax(predict[i]))] += 1
        i += 1
    print('final result: ')
    print(labels)
    print("0", r_0)
    print("1", r_1)
    print("2", r_2)
    print("3", r_3)
    print("4", r_4)
    print("5", r_5)
    print("6", r_6)
    print("7", r_7)
    print("8", r_8)
    print("9", r_9)


def weight_dot(inputs):
    x = inputs[0]
    y = inputs[1]
    return x * y


def compute_same_label(label):
    flag = 0
    count = 0
    for val in label:
        if val > flag:
            flag = val
            count = 0
        elif val == flag:
            count += 1
    return count

# compute acc with regression label
def compute_acc(predict, label):
    accuracy = 0
    count = 0
    # print(predict.shape, label.shape)
    for l in range(len(label)):
        num_l = compute_same_label(label[l])
        num_p = compute_same_label(predict[l])
        if num_l == 0 and num_p == 0:
            if np.argmax(predict[l]) == np.argmax(label[l]):
                accuracy += 1
            count += 1
    # print(count)
    return accuracy / count


# Load the data
train_audio, train_text, test_audio, test_text, train_label, test_label, dev_audio, dev_text, dev_label = get_data(data_path)

# Model Structure
audio_input = Input(shape=(167, 200))
text_input = Input(shape=(167, 200))

fusion_att_a, fusion_att_t = FusionAttention(n_head=n_head, d_k=d_k)([audio_input, text_input])
# fusion_att_a = Dense(128)(fusion_att_a)
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation(activation)(fusion_att_a)
# fusion_att_t = Dense(128)(fusion_att_t)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation(activation)(fusion_att_t)

# fusion_att_a = MaxPooling1D(pool_size=2)(fusion_att_a)
# fusion_att_t = MaxPooling1D(pool_size=2)(fusion_att_t)

fusion_att_a, fusion_att_t = FusionAttention(n_head=n_head, d_k=d_k)([fusion_att_a, fusion_att_t])
# fusion_att_a = Dense(dense_size)(fusion_att_a)
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation(activation)(fusion_att_a)
# fusion_att_t = Dense(dense_size)(fusion_att_t)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation(activation)(fusion_att_t)
#
# fusion_att_a = MaxPooling1D(pool_size=2)(fusion_att_a)
# fusion_att_t = MaxPooling1D(pool_size=2)(fusion_att_t)

fusion_att_a, fusion_att_t = FusionAttention(n_head=n_head, d_k=d_k)([fusion_att_a, fusion_att_t])
# fusion_att_a = Dense(dense_size)(fusion_att_a)
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation(activation)(fusion_att_a)
# fusion_att_t = Dense(dense_size)(fusion_att_t)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation(activation)(fusion_att_t)

# fusion_att_a = MaxPooling1D(pool_size=2)(fusion_att_a)
# fusion_att_t = MaxPooling1D(pool_size=2)(fusion_att_t)
#
fusion_att_a, fusion_att_t = FusionAttention(n_head=n_head, d_k=d_k)([fusion_att_a, fusion_att_t])
# fusion_att_a = Dense(dense_size)(fusion_att_a)
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation(activation)(fusion_att_a)
# fusion_att_t = Dense(dense_size)(fusion_att_t)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation(activation)(fusion_att_t)

# fusion_att_a = MaxPooling1D(pool_size=2)(fusion_att_a)
# fusion_att_t = MaxPooling1D(pool_size=2)(fusion_att_t)

concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))
fusion_att_concat = concat([fusion_att_a, fusion_att_t])

# fusion_att_concat = Dense(256)(fusion_att_concat)
# fusion_att_concat = BatchNormalization()(fusion_att_concat)
# fusion_att_concat = Activation('relu')(fusion_att_concat)
# # fusion_att_concat = Dropout(dropout)(fusion_att_concat)
#
# fusion_att_concat = MaxPooling1D(pool_size=2)(fusion_att_concat)
#
# fusion_att_concat = Dense(128)(fusion_att_concat)
# fusion_att_concat = BatchNormalization()(fusion_att_concat)
# fusion_att_concat = Activation('relu')(fusion_att_concat)
# # fusion_att_concat = Dropout(dropout)(fusion_att_concat)

# fusion_att_concat = MaxPooling1D(pool_size=2)(fusion_att_concat)

fusion_att_concat = Dense(256)(fusion_att_concat)
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('tanh')(fusion_att_concat)
# fusion_att_concat = Dropout(dropout)(fusion_att_concat)

# fusion_att_concat = MaxPooling1D(pool_size=4)(fusion_att_concat)

fusion_att_concat = GlobalMaxPooling1D()(fusion_att_concat)

fusion_att_concat = Dense(64)(fusion_att_concat)
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('tanh')(fusion_att_concat)

# fusion_att_concat = MaxPooling1D(pool_size=4)(fusion_att_concat)

fusion_att_concat = Dense(16)(fusion_att_concat)
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('tanh')(fusion_att_concat)

# fusion_att_concat = MaxPooling1D(pool_size=8)(fusion_att_concat)
# fusion_att_concat = Lambda(remove_dimensions)(fusion_att_concat)

prediction = Dense(10)(fusion_att_concat)
model = Model(inputs=[audio_input, text_input], outputs=prediction)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.000001, momentum=0.0, decay=0.99, nesterov=True)

model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
model.summary()
print(get_flops(model))

# train model
for i in range(size):
    print('audio branch, epoch: ', str(i))
    epo_audio, epo_text, epo_label = shuffle(train_audio, train_text, train_label)
    history = model.fit([epo_audio, epo_text], epo_label, batch_size=batch_size, epochs=1, verbose=1)
    loss_train.append(history.history['loss'])
    loss_f, mae_f = model.evaluate([dev_audio, dev_text], dev_label, batch_size=batch_size, verbose=0)
    mae_test.append(mae_f)
    loss_test.append(loss_f)
    print('epoch: ', str(i))
    print('loss_a', loss_f, ' ', 'mae_a', mae_f)
    if loss_f <= loss:
        mae = mae_f
        loss = loss_f
        output_test = model.predict([test_audio, test_text], batch_size=batch_size)
        acc= compute_acc(output_test, test_label)
        model.save_weights(save_path + 'fusion_context.h5')
        print('loss', loss, 'acc: ', acc)
print('loss', loss, 'mae: ', mae, 'acc', acc)

# test model
model.load_weights(save_path + 'fusion_context.h5')
output_test = model.predict([test_audio, test_text], batch_size=batch_size)
acc = compute_acc(output_test, test_label)
print(acc)
analyze_data(test_label, output_test)
