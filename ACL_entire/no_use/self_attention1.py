#! -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Zeros, Ones
from keras.layers import Dropout, Add, Conv1D


class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000.,
                                2 * K.arange(self.size / 2, dtype='float32'
                                             ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(Layer):


    def __init__(self, nb_head, size_per_head,dropout, attn_dropout,ffn_dropout,use_norm,use_ffn,**kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        self.layer_norm = LayerNormalization() if use_norm else None
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.pos_ffn_layer = PositionwiseFeedForward(200, 800, dropout=ffn_dropout) if use_ffn else None
        super(Attention, self).__init__(**kwargs)


    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.size_per_head),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.size_per_head),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.size_per_head),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WO = self.add_weight(name='WO',
                                  shape=(self.output_dim,input_shape[0][-1]),
                                  initializer='glorot_uniform',
                                  trainable=True
                                  )
        super(Attention, self).build(input_shape)

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        O_seq_list = []
        for i in range(self.nb_head):
            Q_seq, K_seq, V_seq = x
            print(Q_seq.shape[0])
            print(Q_seq.shape[1])
            print(Q_seq.shape[2])
            print(Q_seq.shape[3])
            Q_seq = K.dot(Q_seq, self.WQ)  #(?,50,20)
            K_seq = K.dot(K_seq, self.WK)  #(?,50,20)
            V_seq = K.dot(V_seq, self.WV)  #(?,50,20)

            # 计算内积，然后mask，然后softmax
            A = K.batch_dot(Q_seq, K_seq,axes=[2,2]) / self.size_per_head ** 0.5
            A = K.softmax(A)#(?,50,50)
            A = Dropout(self.attn_dropout)(A)

            # 输出并mask
            O_seq = K.batch_dot(A, V_seq, axes=[2, 1])#(?,50,20)
            O_seq_list.append(O_seq)

        #concat
        O_seq_concat = O_seq_list[0]
        for i in range(1, len(O_seq_list)):
            O_seq_concat = K.concatenate([O_seq_concat, O_seq_list[i]])
        O_seq = K.dot(O_seq_concat,self.WO)
        #outputs = Dropout(self.dropout)(O_seq)
        ''' if not self.layer_norm: return outputs
        outputs = Add()([outputs,x[0]])
        outputs = self.layer_norm(outputs)
        if not self.pos_ffn_layer: return outputs
        return self.pos_ffn_layer(outputs)'''
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][-1])

class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):  # (?,50,200)
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape
