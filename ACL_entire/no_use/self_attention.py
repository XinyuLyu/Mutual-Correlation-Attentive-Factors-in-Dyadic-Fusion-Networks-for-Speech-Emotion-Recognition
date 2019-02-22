from keras.layers import *
from keras.initializers import *
import tensorflow as tf

class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000.,2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

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


class ScaledDotProductAttention():
    def __init__(self, d_k, attn_dropout):
        self.temper = np.sqrt(d_k)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v):# q,k,v:[n_head * batch_size, len_k, d_k](4*?,50,d_k)
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])# attn:[n_head * batch_size, len_k, len_k](4*?,50,50)
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v]) # output:[n_head * batch_size, len_k, len_k](4*?,50,d_k)
        return output

class PositionwiseFeedForward():
    def __init__(self, d_model, ffn_dropout):
        self.w_1 = Conv1D(400, 1, activation='relu')#2048
        self.w_2 = Conv1D(200, 1)#512
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(ffn_dropout)

    def __call__(self, x):  # x:(?,50,200)
        output = self.w_1(x) # output:(?,50,800)
        output = self.w_2(output) # output:(?,50,200)
        output = self.dropout(output)
        output = Add()([output, x]) # output:(?,50,200),(?,50,200)
        return self.layer_norm(output)#return :(?,50,200)

class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_k, d_model, attn_dropout, ffn_dropout, use_norm=True, use_ffn=True):
        self.n_head = n_head
        self.d_k = d_k
        self.attn_dropout = attn_dropout
        self.input_shape = d_model
        self.ffn_dropout = ffn_dropout
        self.use_norm = use_norm
        self.use_ffn = use_ffn

        self.qs_layer = Dense(n_head * d_k, use_bias=False)  # 过WQ
        self.ks_layer = Dense(n_head * d_k, use_bias=False)
        self.vs_layer = Dense(n_head * d_k, use_bias=False)

        self.attention = ScaledDotProductAttention(d_k, attn_dropout)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, ffn_dropout) if use_ffn else None

    def __call__(self, q, k, v):# q,k,v (?,50,200)
        d_k, d_v = self.d_k, self.d_k
        n_head = self.n_head

        qs = self.qs_layer(q)  # qs:[batch_size, len_k, n_head*d_k](?,50,n_head*d_k)
        ks = self.ks_layer(k)  # ks:[batch_size, len_k, n_head*d_k](?,50,n_head*d_k)
        vs = self.vs_layer(v)  # vs:[batch_size, len_k, n_head*d_k](?,50,n_head*d_k)

        def reshape1(x):
            s = tf.shape(x)  # s:[batch_size, len_k, n_head * d_k]
            x = tf.reshape(x, [s[0], s[1], n_head, s[2] // n_head])  # x:[batch_size, len_k, n_head , d_k] (?,50,n_head,d_k)
            x = tf.transpose(x, [2, 0, 1, 3])  # x:[ n_head, batch_size , len_k, d_k]  (?,n_head,50,d_k)
            x = tf.reshape(x, [-1, s[1], s[2] // n_head])  # x:[n_head * batch_size, len_k, d_k](n_head*?,50,d_k)
            return x

        qss = Lambda(reshape1)(qs)  # qss:[n_head * batch_size, len_k, d_k](n_head*?,50,d_k)
        kss = Lambda(reshape1)(ks)  # kss:[n_head * batch_size, len_k, d_k](n_head*?,50,d_k)
        vss = Lambda(reshape1)(vs)  # vss:[n_head * batch_size, len_k, d_k](n_head*?,50,d_k)

        head = self.attention(qss, kss, vss)  # head:[n_head * batch_size, len_k, d_k](n_head*?,50,d_k)

        def reshape2(x):
            s = tf.shape(x)  #s:[n_head * batch_size, len_k, d_k](n_head*?,50,d_k)
            x = tf.reshape(x, [n_head, -1, s[1], s[2]])   # x:[n_head, batch_size,len_k, d_k](n_head,? , 50, d_k)
            x = tf.transpose(x, [1, 2, 0, 3])            # x:[batch_size, len_k, n_head , d_k](?, 50, n_head , d_k)
            x = tf.reshape(x, [-1, s[1], n_head * d_v])  # x:[batch_size, len_k, n_head * d_k](?, 50, n_head*d_k)
            return x

        head = Lambda(reshape2)(head)  # head:[batch_size, len_k, n_head * d_k](?,50,n_head * d_k)

        outputs = self.w_o(head) #outputs:[batch_size, len_k, 200] (?,50,200)
        outputs = Dropout(self.attn_dropout)(outputs)  # outputs:(?,50,200)
        if not self.layer_norm: return outputs
        outputs = Add()([outputs, q])  # outputs:(?,50,200)
        outputs = self.layer_norm(outputs)#outputs:(?,50,200)
        if not self.pos_ffn_layer: return outputs
        return self.pos_ffn_layer(outputs)#return :(?,50,200)


