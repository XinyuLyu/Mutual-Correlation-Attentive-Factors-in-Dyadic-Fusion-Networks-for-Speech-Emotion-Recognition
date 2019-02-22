#! -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Ones, Zeros
from keras.layers import Conv1D, Dropout, Add, TimeDistributed, Dense
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

    def __init__(self, n_head, d_k, **kwargs):
        self.n_head = n_head
        self.d_k = d_k
        self.output_dim = n_head * d_k
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WO = self.add_weight(name='WO',
                                  shape=(self.output_dim,input_shape[0][-1]),
                                  initializer='glorot_uniform',
                                  trainable=True
                                  )

        super(Attention, self).build(input_shape)

    def call(self, x):
        Q_seq, K_seq , V_seq =x
        #qs_layer
        Q_seq1 = K.dot(Q_seq, self.WQ)#(?,50,n_head*d_k)
        K_seq1 = K.dot(K_seq, self.WK)
        V_seq1 = K.dot(V_seq, self.WV)
        #reshape 1
        Q_seq2 = K.reshape(Q_seq1, (-1, K.shape(Q_seq1)[1], self.n_head, self.d_k))#(?,50,n_head,d_k)
        Q_seq3 = K.permute_dimensions(Q_seq2, (0, 2, 1, 3))#(?,n_head,50,d_k)
        K_seq2 = K.reshape(K_seq1, (-1, K.shape(K_seq1)[1], self.n_head, self.d_k))
        K_seq3 = K.permute_dimensions(K_seq2, (0, 2, 1, 3))
        V_seq2 = K.reshape(V_seq1, (-1, K.shape(V_seq1)[1], self.n_head, self.d_k))   # 10，20
        V_seq3 = K.permute_dimensions(V_seq2, (0, 2, 1, 3))

        # attention
        A = K.batch_dot(Q_seq3, K_seq3, axes=[3, 3]) / self.d_k ** 0.5#(?,n_head,50,50)
        A1 = K.softmax(A)
        #A2 = Dropout(0.1)(A1)

        # reshape 2
        O_seq = K.batch_dot(A1, V_seq3, axes=[3, 2])#(?,n_head,50,d_k)
        O_seq1 = K.permute_dimensions(O_seq, (0, 2, 1, 3))#(?,50,n_head,d_k)
        O_seq2 = K.reshape(O_seq1, [-1, K.shape(O_seq1)[1], self.output_dim])  # (?, 50, n_head*d_k)

        #w_o
        outputs = K.dot(O_seq2, self.WO)# (?, 50, 64)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class FusionAttention(Layer):

    def __init__(self, n_head, d_k, **kwargs):
        self.n_head = n_head
        self.d_k = d_k
        self.output_dim = n_head * d_k

        super(FusionAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ_a = self.add_weight(name='WQ_a',
                                    shape=(input_shape[0][-1], self.output_dim),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.WK_a = self.add_weight(name='WK_a',
                                    shape=(input_shape[1][-1], self.output_dim),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.WV_a = self.add_weight(name='WV_a',
                                    shape=(input_shape[2][-1], self.output_dim),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.WQ_t = self.add_weight(name='WQ_t',
                                    shape=(input_shape[3][-1], self.output_dim),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.WK_t = self.add_weight(name='WK_t',
                                    shape=(input_shape[4][-1], self.output_dim),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.WV_t = self.add_weight(name='WV_t',
                                    shape=(input_shape[5][-1], self.output_dim),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.WO_a = self.add_weight(name='WO_a',
                                    shape=(self.output_dim, input_shape[0][-1]),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.WO_t = self.add_weight(name='WO_t',
                                    shape=(self.output_dim, input_shape[3][-1]),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.WF_a = self.add_weight(name='WF_a',
                                    shape=(self.d_k, self.d_k),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.WF_t = self.add_weight(name='WF_t',
                                    shape=(self.d_k, self.d_k),
                                    initializer='glorot_uniform',
                                    trainable=True)

        super(FusionAttention, self).build(input_shape)

    def call(self, x):
        q_a, k_a, v_a, q_t, k_t, v_t = x

        # q, k, v layers for text and audio branch
        q_a = K.dot(q_a, self.WQ_a)
        k_a = K.dot(k_a, self.WK_a)
        v_a = K.dot(v_a, self.WV_a)
        q_t = K.dot(q_t, self.WQ_t)
        k_t = K.dot(k_t, self.WK_t)
        v_t = K.dot(v_t, self.WV_t)

        # multi-head reshape
        q_a = K.reshape(q_a, (-1, K.shape(q_a)[1], self.n_head, self.d_k))
        q_a = K.permute_dimensions(q_a, (0, 2, 1, 3))
        k_a = K.reshape(k_a, (-1, K.shape(k_a)[1], self.n_head, self.d_k))
        k_a = K.permute_dimensions(k_a, (0, 2, 1, 3))
        v_a = K.reshape(v_a, (-1, K.shape(v_a)[1], self.n_head, self.d_k))
        v_a = K.permute_dimensions(v_a, (0, 2, 1, 3))

        q_t = K.reshape(q_t, (-1, K.shape(q_t)[1], self.n_head, self.d_k))
        q_t = K.permute_dimensions(q_t, (0, 2, 1, 3))
        k_t = K.reshape(k_t, (-1, K.shape(k_t)[1], self.n_head, self.d_k))
        k_t = K.permute_dimensions(k_t, (0, 2, 1, 3))
        v_t = K.reshape(v_t, (-1, K.shape(v_t)[1], self.n_head, self.d_k))
        v_t = K.permute_dimensions(v_t, (0, 2, 1, 3))

        # fusion factor for k_a and k_t
        k_a = K.reshape(k_a, (-1, K.shape(k_a)[2], self.d_k))
        k_t = K.reshape(k_t, (-1, K.shape(k_t)[2], self.d_k))
        k_a = K.dot(k_a, self.WF_a) + K.dot(k_t, self.WF_a)
        k_t = K.dot(k_a, self.WF_t) + K.dot(k_t, self.WF_t)
        k_a = K.reshape(k_a, (-1, self.n_head, K.shape(k_a)[1], self.d_k))
        k_t = K.reshape(k_t, (-1, self.n_head, K.shape(k_t)[1], self.d_k))

        # attention
        score_a = K.batch_dot(q_a, k_a, axes=[3, 3]) / self.d_k ** 0.5
        score_a = K.softmax(score_a)
        o_a = K.batch_dot(score_a, v_a, axes=[3, 2])
        score_t = K.batch_dot(q_t, k_t, axes=[3, 3]) / self.d_k ** 0.5
        score_t = K.softmax(score_t)
        o_t = K.batch_dot(score_t, v_t, axes=[3, 2])

        # concatenate the multi-heads
        o_a = K.permute_dimensions(o_a, (0, 2, 1, 3))
        o_a = K.reshape(o_a, [-1, K.shape(o_a)[1], self.output_dim])
        o_t = K.permute_dimensions(o_t, (0, 2, 1, 3))
        o_t = K.reshape(o_t, [-1, K.shape(o_t)[1], self.output_dim])

        # dense layer
        outputs_a = K.dot(o_a, self.WO_a)
        outputs_t = K.dot(o_t, self.WO_t)

        return [outputs_a, outputs_t]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0]), (input_shape[3])]#

