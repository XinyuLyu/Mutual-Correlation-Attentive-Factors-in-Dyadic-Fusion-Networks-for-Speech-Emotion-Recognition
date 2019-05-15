#! -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Ones, Zeros

init = 'he_uniform'
trainable = True


class FusionAttention(Layer):

    def __init__(self, n_head, d_k, **kwargs):
        self.n_head = n_head
        self.d_k = d_k
        self.output_dim = n_head * d_k

        super(FusionAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ_a = self.add_weight(name='WQ_a',
                                    shape=(input_shape[0][-1], self.output_dim),
                                    initializer=init,
                                    trainable=True)
        self.WK_a = self.add_weight(name='WK_a',
                                    shape=(input_shape[0][-1], self.output_dim),
                                    initializer=init,
                                    trainable=True)
        self.WV_a = self.add_weight(name='WV_a',
                                    shape=(input_shape[0][-1], self.output_dim),
                                    initializer=init,
                                    trainable=True)
        self.WO_a = self.add_weight(name='WO_a',
                                    shape=(self.output_dim, input_shape[1][-1]),
                                    initializer=init,
                                    trainable=True)
        self.WQ_t = self.add_weight(name='WQ_t',
                                    shape=(input_shape[1][-1], self.output_dim),
                                    initializer=init,
                                    trainable=trainable)
        self.WK_t = self.add_weight(name='WK_t',
                                    shape=(input_shape[1][-1], self.output_dim),
                                    initializer=init,
                                    trainable=trainable)
        self.WV_t = self.add_weight(name='WV_t',
                                    shape=(input_shape[1][-1], self.output_dim),
                                    initializer=init,
                                    trainable=trainable)
        self.WO_t = self.add_weight(name='WO_t',
                                    shape=(self.output_dim, input_shape[1][-1]),
                                    initializer=init,
                                    trainable=trainable)

        self.WF_t = self.add_weight(name='WF_t',
                                    shape=(self.d_k, self.d_k),
                                    initializer=init,
                                    trainable=trainable)
        self.WF_a = self.add_weight(name='WF_a',
                                    shape=(self.d_k, self.d_k),
                                    initializer=init,
                                    trainable=trainable)

        super(FusionAttention, self).build(input_shape)

    def call(self, x):
        a, t = x
        q_a, k_a, v_a = a, a, a  # （?，167，200）
        q_t, k_t, v_t = t, t, t

        # q, k, v layers for text and audio branch
        q_a = K.dot(q_a, self.WQ_a)  # （?，167，200）
        k_a = K.dot(k_a, self.WK_a)
        v_a = K.dot(v_a, self.WV_a)
        q_t = K.dot(q_t, self.WQ_t)
        k_t = K.dot(k_t, self.WK_t)
        v_t = K.dot(v_t, self.WV_t)

        # multi-head reshape
        q_a = K.reshape(q_a, (-1, K.shape(q_a)[1], self.n_head, self.d_k))  # (?,167,10,20)
        q_a = K.permute_dimensions(q_a, (0, 2, 1, 3))  # (?,10,167,20)
        k_a = K.reshape(k_a, (-1, K.shape(k_a)[1], self.n_head, self.d_k))
        k_a = K.permute_dimensions(k_a, (0, 2, 1, 3))
        v_a = K.reshape(v_a, (-1, K.shape(v_a)[1], self.n_head, self.d_k))
        v_a = K.permute_dimensions(v_a, (0, 2, 1, 3))

        q_t = K.reshape(q_t, (-1, K.shape(q_t)[1], self.n_head, self.d_k))  # (?,167,10,20)
        q_t = K.permute_dimensions(q_t, (0, 2, 1, 3))  # (?,10,167,20)
        k_t = K.reshape(k_t, (-1, K.shape(k_t)[1], self.n_head, self.d_k))
        k_t = K.permute_dimensions(k_t, (0, 2, 1, 3))
        v_t = K.reshape(v_t, (-1, K.shape(v_t)[1], self.n_head, self.d_k))
        v_t = K.permute_dimensions(v_t, (0, 2, 1, 3))# (?,10,167,20)

        # fusion factor for k_a and k_t
        k_a = K.reshape(k_a, (-1, K.shape(k_a)[2], self.d_k))  # (?*10,167,20)
        k_t = K.reshape(k_t, (-1, K.shape(k_t)[2], self.d_k))
        # k_a = K.dot(k_a, self.WF_a) + K.dot(k_t, self.WF_a)
        # k_t = K.dot(k_a, self.WF_t) + K.dot(k_t, self.WF_t)
        k_a = k_a + K.dot(k_t, self.WF_a)
        k_t = k_t + K.dot(k_a, self.WF_t)
        k_a = K.reshape(k_a, (-1, self.n_head, K.shape(k_a)[1], self.d_k))
        k_t = K.reshape(k_t, (-1, self.n_head, K.shape(k_t)[1], self.d_k))  # (?,10,167,20)

        # attention
        score_a = K.batch_dot(q_a, k_a, axes=[3, 3]) / self.d_k ** 0.5  # (?,10,167,167)
        score_a = K.softmax(score_a)
        o_a = K.batch_dot(score_a, v_a, axes=[3, 2])
        score_t = K.batch_dot(q_t, k_t, axes=[3, 3]) / self.d_k ** 0.5  # (?,10,167,167)
        score_t = K.softmax(score_t)
        o_t = K.batch_dot(score_t, v_t, axes=[3, 2])  # (?,10,167,20)

        # concatenate the multi-heads
        o_a = K.permute_dimensions(o_a, (0, 2, 1, 3))  # (?,50,4,16)
        o_a = K.reshape(o_a, [-1, K.shape(o_a)[1], self.output_dim])  # (?,50,64)
        o_t = K.permute_dimensions(o_t, (0, 2, 1, 3))
        o_t = K.reshape(o_t, [-1, K.shape(o_t)[1], self.output_dim])

        # dense layer
        output_a = K.dot(o_a, self.WO_a)  # (?,50,200)
        output_t = K.dot(o_t, self.WO_t)

        return [output_a, output_t, score_a, score_t]

    def compute_output_shape(self, input_shape):  # (?，167，200）=> (?,10,167,167)
        return [(input_shape[1]),
                (input_shape[1]),
                (input_shape[1][0], self.n_head, input_shape[1][1], input_shape[1][1]),
                (input_shape[1][0], self.n_head, input_shape[1][1], input_shape[1][1])]

        # return [ (input_shape[1]),(input_shape[1]),
        #         (-1, self.n_head, input_shape[1][1], input_shape[1][1]),
        #         (-1, self.n_head, input_shape[1][1], input_shape[1][1])]


class Attention(Layer):

    def __init__(self, n_head, d_k, **kwargs):
        self.n_head = n_head
        self.d_k = d_k
        self.output_dim = n_head * d_k

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ_a = self.add_weight(name='WQ_a',
                                    shape=(input_shape[0][-1], self.output_dim),
                                    initializer=init,
                                    trainable=True)
        self.WK_a = self.add_weight(name='WK_a',
                                    shape=(input_shape[0][-1], self.output_dim),
                                    initializer=init,
                                    trainable=True)
        self.WV_a = self.add_weight(name='WV_a',
                                    shape=(input_shape[0][-1], self.output_dim),
                                    initializer=init,
                                    trainable=True)
        self.WO_a = self.add_weight(name='WO_a',
                                    shape=(self.output_dim, input_shape[1][-1]),
                                    initializer=init,
                                    trainable=True)
        self.WQ_t = self.add_weight(name='WQ_t',
                                    shape=(input_shape[1][-1], self.output_dim),
                                    initializer=init,
                                    trainable=trainable)
        self.WK_t = self.add_weight(name='WK_t',
                                    shape=(input_shape[1][-1], self.output_dim),
                                    initializer=init,
                                    trainable=trainable)
        self.WV_t = self.add_weight(name='WV_t',
                                    shape=(input_shape[1][-1], self.output_dim),
                                    initializer=init,
                                    trainable=trainable)
        self.WO_t = self.add_weight(name='WO_t',
                                    shape=(self.output_dim, input_shape[1][-1]),
                                    initializer=init,
                                    trainable=trainable)

        self.WF_t = self.add_weight(name='WF_t',
                                    shape=(self.d_k, self.d_k),
                                    initializer=init,
                                    trainable=trainable)
        self.WF_a = self.add_weight(name='WF_a',
                                    shape=(self.d_k, self.d_k),
                                    initializer=init,
                                    trainable=trainable)

        super(Attention, self).build(input_shape)

    def call(self, x):
        a, t = x
        q_a, k_a, v_a = a, a, a  # （?，167，200）
        q_t, k_t, v_t = t, t, t

        # q, k, v layers for text and audio branch
        q_a = K.dot(q_a, self.WQ_a)  # （?，167，200）
        k_a = K.dot(k_a, self.WK_a)
        v_a = K.dot(v_a, self.WV_a)
        q_t = K.dot(q_t, self.WQ_t)
        k_t = K.dot(k_t, self.WK_t)
        v_t = K.dot(v_t, self.WV_t)

        # multi-head reshape
        q_a = K.reshape(q_a, (-1, K.shape(q_a)[1], self.n_head, self.d_k))  # (?,167,10,20)
        q_a = K.permute_dimensions(q_a, (0, 2, 1, 3))  # (?,10,167,20)
        k_a = K.reshape(k_a, (-1, K.shape(k_a)[1], self.n_head, self.d_k))
        k_a = K.permute_dimensions(k_a, (0, 2, 1, 3))
        v_a = K.reshape(v_a, (-1, K.shape(v_a)[1], self.n_head, self.d_k))
        v_a = K.permute_dimensions(v_a, (0, 2, 1, 3))

        q_t = K.reshape(q_t, (-1, K.shape(q_t)[1], self.n_head, self.d_k))  # (?,167,10,20)
        q_t = K.permute_dimensions(q_t, (0, 2, 1, 3))  # (?,10,167,20)
        k_t = K.reshape(k_t, (-1, K.shape(k_t)[1], self.n_head, self.d_k))
        k_t = K.permute_dimensions(k_t, (0, 2, 1, 3))
        v_t = K.reshape(v_t, (-1, K.shape(v_t)[1], self.n_head, self.d_k))
        v_t = K.permute_dimensions(v_t, (0, 2, 1, 3))

        # fusion factor for k_a and k_t
        k_a = K.reshape(k_a, (-1, K.shape(k_a)[2], self.d_k))  # (?*10,167,20)
        k_t = K.reshape(k_t, (-1, K.shape(k_t)[2], self.d_k))
        # k_a = K.dot(k_a, self.WF_a) + K.dot(k_t, self.WF_a)
        # k_t = K.dot(k_a, self.WF_t) + K.dot(k_t, self.WF_t)
        k_a = k_a + K.dot(k_t, self.WF_a)
        k_t = k_t + K.dot(k_a, self.WF_t)
        k_a = K.reshape(k_a, (-1, self.n_head, K.shape(k_a)[1], self.d_k))
        k_t = K.reshape(k_t, (-1, self.n_head, K.shape(k_t)[1], self.d_k))  # (?,10,167,20)

        # attention
        score_a = K.batch_dot(q_a, k_a, axes=[3, 3]) / self.d_k ** 0.5  # (?,10,167,167)
        score_a = K.softmax(score_a)
        V_a = v_a # (?,10,167,20)
        o_a = K.batch_dot(score_a, v_a, axes=[3, 2])
        score_t = K.batch_dot(q_t, k_t, axes=[3, 3]) / self.d_k ** 0.5  # (?,10,167,167)
        score_t = K.softmax(score_t)
        V_t = v_t # (?,10,167,20)
        o_t = K.batch_dot(score_t, v_t, axes=[3, 2])  # (?,10,167,20)
        O_t1 = o_t

        # concatenate the multi-heads
        o_a = K.permute_dimensions(o_a, (0, 2, 1, 3))  # (?,167,10,20)
        o_a = K.reshape(o_a, [-1, K.shape(o_a)[1], self.output_dim])  # (?,167,200)
        o_t = K.permute_dimensions(o_t, (0, 2, 1, 3))
        o_t = K.reshape(o_t, [-1, K.shape(o_t)[1], self.output_dim])
        O_t2 = o_t
        # dense layer
        output_a = K.dot(o_a, self.WO_a)  # (?,50,200)
        output_t = K.dot(o_t, self.WO_t)

        return [output_a, output_t, O_t1, O_t2]

    def compute_output_shape(self, input_shape):  # (?，167，200）=> (?,10,167,167)
        return [(input_shape[1]),
                (input_shape[1]),
                (input_shape[1][0], self.n_head, input_shape[1][1], 20),
                (input_shape[1][0], self.n_head, input_shape[1][1]* 20)]

        # return [ (input_shape[1]),(input_shape[1]),
        #         (-1, self.n_head, input_shape[1][1], input_shape[1][1]),
        #         (-1, self.n_head, input_shape[1][1], input_shape[1][1])]
