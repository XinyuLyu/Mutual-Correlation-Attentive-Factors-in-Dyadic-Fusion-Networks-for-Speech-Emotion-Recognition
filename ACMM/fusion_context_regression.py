from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import MaxPooling1D
from keras import backend as K
from keras import regularizers as rl
from keras import Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from Document_level_analysis.self_attention_fusion import FusionAttention
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import numpy as np

data_path = r'E:\Yue\Code\ACL_entire\Document_level_analysis\fusion_context\\'
save_path = ''
batch_size = 16
n_head = 10
d_k = 20
activation = 'relu'
dense_size = 64

acc = 0
mae = 100
loss = 1000
loss_train = []
mae_train = []
loss_test = []
mae_test = []
size = 300
epoch = np.linspace(1, size, size)


def get_data(path):
    print('loading data...')
    train_a = np.load(path + 'audio_train.npy')
    test_a = np.load(path + 'audio_test.npy')
    train_t = np.load(path + 'text_train.npy')
    test_t = np.load(path + 'text_test.npy')
    train_l = np.load(path + 'train_label.npy')
    test_l = np.load(path + 'test_label.npy')
    print('finish loading data...')

    return train_a, train_t, test_a, test_t, train_l, test_l


def expand_dimensions(x):
    return K.expand_dims(x)


def remove_dimensions(x):
    return K.squeeze(x, axis=1)


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
train_audio, train_text, test_audio, test_text, train_label, test_label = get_data(data_path)

# Model Structure
audio_input = Input(shape=(167, 200))
text_input = Input(shape=(167, 200))

# Model Structure
audio_input = Input(shape=(167, 200))
text_input = Input(shape=(167, 200))

# audio_rep = Dense(dense_size)(audio_input)
# audio_rep = BatchNormalization()(audio_rep)
# audio_rep = Activation(activation)(audio_rep)
# text_rep = Dense(dense_size)(text_input)
# text_rep = BatchNormalization()(text_rep)
# text_rep = Activation(activation)(text_rep)

fusion_att_a, fusion_att_t = FusionAttention(n_head=n_head, d_k=d_k)([audio_input, text_input])
fusion_att_a = Dense(dense_size)(fusion_att_a)
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation(activation)(fusion_att_a)
fusion_att_t = Dense(dense_size)(fusion_att_t)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation(activation)(fusion_att_t)

fusion_att_a, fusion_att_t = FusionAttention(n_head=n_head, d_k=d_k)([fusion_att_a, fusion_att_t])
fusion_att_a = Dense(dense_size)(fusion_att_a)
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation(activation)(fusion_att_a)
fusion_att_t = Dense(dense_size)(fusion_att_t)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation(activation)(fusion_att_t)

fusion_att_a, fusion_att_t = FusionAttention(n_head=n_head, d_k=d_k)([fusion_att_a, fusion_att_t])
fusion_att_a = Dense(dense_size)(fusion_att_a)
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation(activation)(fusion_att_a)
fusion_att_t = Dense(dense_size)(fusion_att_t)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation(activation)(fusion_att_t)

fusion_att_a, fusion_att_t = FusionAttention(n_head=n_head, d_k=d_k)([fusion_att_a, fusion_att_t])
fusion_att_a = Dense(dense_size)(fusion_att_a)
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation(activation)(fusion_att_a)
fusion_att_t = Dense(dense_size)(fusion_att_t)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation(activation)(fusion_att_t)

concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))
fusion_att_concat = concat([fusion_att_a, fusion_att_t])


fusion_att_concat = Dense(256)(fusion_att_concat)
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('relu')(fusion_att_concat)

fusion_att_concat = MaxPooling1D(pool_size=4)(fusion_att_concat)

fusion_att_concat = Dense(128)(fusion_att_concat)
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('relu')(fusion_att_concat)

fusion_att_concat = MaxPooling1D(pool_size=4)(fusion_att_concat)

fusion_att_concat = Dense(64)(fusion_att_concat)
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('relu')(fusion_att_concat)

fusion_att_concat = MaxPooling1D(pool_size=2)(fusion_att_concat)

fusion_att_concat = Dense(32)(fusion_att_concat)
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('relu')(fusion_att_concat)

fusion_att_concat = MaxPooling1D(pool_size=2)(fusion_att_concat)

fusion_att_concat = Dense(16)(fusion_att_concat)
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('relu')(fusion_att_concat)

fusion_att_concat = MaxPooling1D(pool_size=2)(fusion_att_concat)
fusion_att_concat = Lambda(remove_dimensions)(fusion_att_concat)

prediction = Dense(10)(fusion_att_concat)
model = Model(inputs=[audio_input, text_input], outputs=prediction)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.000001, momentum=0.0, decay=0.99, nesterov=True)

# model.load_weights(save_path + 'fusion_context_regression.h5')

model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
model.summary()


# Sentence-level
for i in range(size):
    print('audio branch, epoch: ', str(i))
    epo_audio, epo_text, epo_label = shuffle(train_audio, train_text, train_label)
    history = model.fit([epo_audio, epo_text], epo_label, batch_size=batch_size, epochs=1, verbose=1)
    loss_train.append(history.history['loss'])
    loss_f, mae_f = model.evaluate([test_audio, test_text], test_label, batch_size=batch_size, verbose=0)
    output_test = model.predict([test_audio, test_text], batch_size=batch_size)
    acc_f = compute_acc(output_test, test_label)
    mae_test.append(mae_f)
    loss_test.append(loss_f)
    print('epoch: ', str(i))
    print('loss_a', loss_f, ' ', 'mae_a', mae_f, 'acc', acc_f)
    if mae_f <= mae:
        mae = mae_f
        loss = loss_f
        acc = acc_f
        model.save_weights(save_path + 'fusion_context_regression.h5')
print('loss', loss, 'mae: ', mae, 'acc', acc)

plt.figure()
plt.plot(epoch, loss_train, label='loss_train')
plt.plot(epoch, loss_test, label='loss_test')
plt.plot(epoch, mae_test, label='mae_test')
plt.xlabel("epoch")
plt.ylabel("loss and mae")
plt.legend()
plt.show()
