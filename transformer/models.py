import keras
import keras.backend as K
import tensorflow as tf
import keras.layers as layers
from keras.callbacks import Callback
from keras.models import Model
import transformer.constants as Const
import numpy as np
import math

class ScaledDotProductAttention():
    ''' Scaled Dot Product Attention module in Transformer '''
    def __init__(self, discount_rate, attn_dropout=0.1):
        self.discount_rate = discount_rate
        self.dropout = layers.Dropout(attn_dropout)
        self.softmax = layers.Softmax(axis=-1)
    def __call__(self, keys, query, values, mask=None):
        attention = layers.Dot(axes=2)([query, keys])
        attention = layers.Lambda(lambda x: x / math.sqrt(self.discount_rate))(attention)
        if mask is not None:
            attention = layers.Multiply()([attention, mask])
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        out = layers.Dot(axes=(2,1))([attention, values])
        return out, attention
    
class LayerNormalization(layers.Layer):
    '''
    https://github.com/CyberZHG/keras-layer-normalization/blob/master/keras_layer_normalization/layer_normalization.py
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        self.gamma, self.beta = None, None
        super(LayerNormalization, self).__init__(*kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask
    
    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=input_shape[-1:],
            initializer=keras.initializers.ones(),
            name='gamma',
            trainable=True
        )
        self.beta = self.add_weight(
            shape=input_shape[-1:],
            initializer=keras.initializers.zeros(),
            name='beta',
            trainable=True
        )
        super(LayerNormalization, self).build(input_shape)
    
    def call(self, x, mask=None):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + K.epsilon()) + self.beta


class MultiHeadAttention():
    def __init__(self, n_heads, d_model, d_keys, d_values, dropout=0.1, use_norm=True):
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_key = d_keys
        self.d_values = d_values
        self.use_norm = use_norm
        self.q_head_layers = [layers.Dense(d_keys, use_bias=False) for _ in range(n_heads)]
        self.k_head_layers = [layers.Dense(d_keys, use_bias=False) for _ in range(n_heads)]
        self.v_head_layers = [layers.Dense(d_values, use_bias=False) for _ in range(n_heads)]
        self.attention = ScaledDotProductAttention(discount_rate=d_keys)
        self.normalization = LayerNormalization()
        self.fc = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)
    def __call__(self, queries, keys, values, mask=None):
        head_outs = []
        attentions = []
        for idx in range(self.n_heads):
            projection_queries = self.q_head_layers[idx](queries)
            projection_keys = self.k_head_layers[idx](keys)
            projection_values = self.v_head_layers[idx](values)
            head_output, attns = self.attention(projection_keys, projection_queries, projection_values, mask)
            head_outs.append(head_output)
            attentions.append(attns)
        heads = layers.Concatenate(axis=-1)(head_outs)
        attn = layers.Concatenate(axis=-1)(attentions)

        output = self.fc(heads)
        output = self.dropout(output)

        output = layers.Add()([output, queries])
        if self.use_norm:
            output = self.normalization(output)
        return output, attn

class PositionwiseFeedForward():
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cnn_1 = layers.Conv1D(filters=hidden_dim,
                                    kernel_size=1,
                                    strides=1,
                                    activation='relu')
        self.cnn_2 = layers.Conv1D(filters=input_dim,
                                    kernel_size=1,
                                    strides=1,
                                    activation=None)
        self.layernorm = LayerNormalization()
        self.dropout = layers.Dropout(dropout)
    def __call__(self, x):
        residual = x
        output = self.cnn_1(x)
        output = self.cnn_2(output)
        output = layers.Add()([output, residual])
        output = self.layernorm(output)
        return output

class EncoderLayer():
    '''
    Encoder contains self-attention layer and positionwise feedforward layer
    '''
    def __init__(self, d_model, inner_dim, n_heads, d_keys, d_values, dropout=0.1):
        self.self_attention = MultiHeadAttention(n_heads, 
                                                    d_model,
                                                    d_keys,
                                                    d_values,
                                                    dropout=dropout,
                                                    use_norm=True)
        self.positionwise_ff = PositionwiseFeedForward(d_model, inner_dim, dropout=dropout)
    def __call__(self, enc_input, non_pad_mask=None, self_attn_mask=None):
        enc_output, enc_slf_attn = self.self_attention(enc_input,
                                                        enc_input,
                                                        enc_input,
                                                        mask=self_attn_mask)
        if non_pad_mask is not None:
            enc_output = layers.Multiply()([enc_output, non_pad_mask])

        enc_output = self.positionwise_ff(enc_output)
        if non_pad_mask is not None:
            enc_output = layers.Multiply()([enc_output, non_pad_mask])

        return enc_output, enc_slf_attn

class DecoderLayer():
    def __init__(self, d_model, inner_dim, n_heads, d_keys, d_values, dropout=0.1):
        self.self_attention = MultiHeadAttention(n_heads,
                                                    d_model,
                                                    d_keys,
                                                    d_values,
                                                    dropout=dropout,
                                                    use_norm=True)
        self.enc_attention = MultiHeadAttention(n_heads,
                                                    d_model,
                                                    d_keys,
                                                    d_values,
                                                    dropout=dropout,
                                                    use_norm=True)
        self.positionwise_ff = PositionwiseFeedForward(d_model, inner_dim, dropout=dropout)
    
    def __call__(self, dec_input, enc_output,
                    non_pad_mask=None,
                    self_attn_mask=None,
                    dec_enc_attn_mask=None):
        dec_output, dec_self_attn = self.self_attention(dec_input,
                                                        dec_input,
                                                        dec_input,
                                                        mask=self_attn_mask)
        if non_pad_mask is not None:
            dec_output = layers.Multiply()([dec_output, non_pad_mask])
        dec_output, dec_enc_attn = self.enc_attention(dec_output,
                                                        enc_output,
                                                        enc_output,
                                                        mask=dec_enc_attn_mask)
        if non_pad_mask is not None:
            dec_output = layers.Multiply()([dec_output, non_pad_mask])
        dec_output = self.positionwise_ff(dec_output)
        if non_pad_mask is not None:
            dec_output = layers.Multiply()([dec_output, non_pad_mask])
        return dec_output, dec_self_attn, dec_enc_attn

def get_sub_mask(sequences):
    '''
    sequences should have shape of (batch_size, sequence_length, dim)
    mask should have shape of (batch_size, sequence_length, sequence_length)
    if sequence_length = 3 then mask should be as
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]] x batch_size
    '''
    seq_len = tf.shape(sequences)[1]
    batch_shape = tf.shape(sequences)[:1]
    mask = K.cumsum(tf.eye(seq_len, batch_shape=batch_shape), 1)
    return mask

def get_non_pad_mask(source):
    '''
    source: input tensor of shape (batchsize, sequence_length)
    output shape: (batchsize, sequence_length, 1)
    '''
    mask = K.cast(K.expand_dims(K.not_equal(source, Const.PAD), -1), 'float32')
    #mask = K.expand_dims(mask, axis=-1)
    return mask

def get_attn_key_pad_mask(args):
    '''
    key_sequence shape: (batchsize, key_length)
    query_sequence shape: (batchsize, query_length)
    Create mask for padding from key_sequence while doing attention
    Output shape should be (batchsize, query_length, key_length)
    '''
    key_sequence, query_sequence = args
    ones = K.expand_dims(K.ones_like(query_sequence, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(key_sequence, Const.PAD), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2,1])
    return mask

def get_sin_cos_possition_embedding(no_position, d_model, padding_idx=None):
    '''
    Sinusoid position encoding table
    '''
    def angle(position, dim_i):
        #dim 2i and dim 2i+1 have same angle
        return position / np.power(1e4, 2 * (dim_i // 2) / d_model)
    def positon_angle_vector(position):
        return [angle(position, dim_i) for dim_i in range(d_model)]
    
    sinusoid_table = np.array([positon_angle_vector(position) for position in range(no_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) #dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) #dim 2i+1
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0
    return sinusoid_table

class Encoder():
    def __init__(self, src_vocab_size, len_max_src_seq, n_layers, n_heads,
                        d_keys, d_values, d_model, inner_dim, dropout=0.1):
        n_position = len_max_src_seq + 1
        self.d_model = d_model
        self.src_word_emb = layers.Embedding(input_dim=src_vocab_size,
                                                output_dim=d_model,
                                                mask_zero=False)
        self.src_pos_emb = layers.Embedding(input_dim=n_position,
                                                output_dim=d_model,
                                                trainable=False,
                                                mask_zero=False,
                                                weights=[get_sin_cos_possition_embedding(n_position, d_model, padding_idx=0)])
        self.emb_dropout = layers.Dropout(dropout)
        self.layer_stack = [EncoderLayer(d_model,
                                            inner_dim,
                                            n_heads,
                                            d_keys,
                                            d_values,
                                            dropout) for _ in range(n_layers)]
    
    def __call__(self, src_seq, src_pos, return_attn=False):
        # To store attention score
        attns = []

        # Embedding
        word_emb = self.src_word_emb(src_seq)
        pos_emb = self.src_pos_emb(src_pos)
        enc_output = layers.Add()([word_emb, pos_emb])
        enc_output = self.emb_dropout(enc_output)

        # Prepare maskes
        enc_non_pad_mask = layers.Lambda(get_non_pad_mask)(src_seq)
        self_attn_pad_mask = layers.Lambda(get_attn_key_pad_mask)([src_seq, src_seq])
        
        for enc_layer in self.layer_stack:
            # enc_output, enc_slf_attn = enc_layer(enc_output,
            #                                     non_pad_mask=enc_non_pad_mask,
            #                                     self_attn_mask=self_attn_pad_mask)
            enc_output, enc_slf_attn = enc_layer(enc_output,
                                                    self_attn_mask=self_attn_pad_mask)
            if return_attn:
                attns.append(enc_slf_attn)
        if return_attn:
            return enc_output, attns
        return enc_output,

class Decoder():
    def __init__(self, tgt_vocab_size, len_max_tgt_seq, n_layers, n_heads,
                        d_keys, d_values, d_model, inner_dim, dropout=0.1):
        n_postions = len_max_tgt_seq + 1
        self.d_model = d_model
        self.tgt_word_emb = layers.Embedding(input_dim=tgt_vocab_size,
                                                output_dim=d_model,
                                                mask_zero=False)
        self.tgt_pos_emb = layers.Embedding(input_dim=n_postions,
                                                output_dim=d_model,
                                                trainable=False,
                                                mask_zero=False,
                                                weights=[get_sin_cos_possition_embedding(n_postions, d_model, padding_idx=0)])
        self.emb_dropout = layers.Dropout(dropout)
        self.layer_stack = [DecoderLayer(d_model,
                                            inner_dim,
                                            n_heads,
                                            d_keys,
                                            d_values,
                                            dropout=dropout) for _ in range(n_layers)]
    def __call__(self, src_seq, src_pos, tgt_seq, tgt_pos, enc_output, return_atten=False):
        def tensor_greater_than_zero(x):
            return K.cast(K.greater(x, 0), 'float32')
        # To store attention score
        self_attns = []
        dec_enc_attns = []       

        # Embedding
        dec_word_emb = self.tgt_word_emb(tgt_seq)
        dec_pos_emb = self.tgt_pos_emb(tgt_pos)
        dec_output = layers.Add()([dec_word_emb, dec_pos_emb])

        # Prepare maskes
        dec_non_pad_mask = layers.Lambda(get_non_pad_mask)(tgt_seq)
        dec_slf_attn_subseq_mask = layers.Lambda(lambda x: get_sub_mask(sequences=x))(tgt_seq)
    
        dec_slf_attn_keypad_mask = layers.Lambda(get_attn_key_pad_mask)([tgt_seq, tgt_seq])
        dec_slf_attn_mask = layers.Add()([dec_slf_attn_subseq_mask, dec_slf_attn_keypad_mask])
        dec_slf_attn_mask = layers.Lambda(lambda x: tensor_greater_than_zero(x))(dec_slf_attn_mask)

        dec_enc_attn_mask = layers.Lambda(get_attn_key_pad_mask)([src_seq, tgt_seq])

        for dec_layer in self.layer_stack:
            # dec_output, dec_self_attn, dec_enc_attn = dec_layer(dec_output, enc_output,
            #                                                         non_pad_mask=dec_non_pad_mask,
            #                                                         self_attn_mask=dec_slf_attn_mask,
            #                                                         dec_enc_attn_mask=dec_enc_attn_mask)
            dec_output, dec_self_attn, dec_enc_attn = dec_layer(dec_output, enc_output,
                                                                self_attn_mask=dec_slf_attn_mask,
                                                                dec_enc_attn_mask=dec_enc_attn_mask)
            if return_atten:
                self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)
        if return_atten:
            return dec_output, self_attns, dec_enc_attns
        return dec_output,

class Transformer():
    '''
    Transformer model
    '''
    def __init__(self,
                src_vocab_size, tgt_vocab_size, len_max_seq,
                d_model=512, inner_dim=2048,
                n_layers=6, n_heads=8, d_keys=64, d_values=64,
                dropout=0.1, tgt_pred_emb_share_weights=True,
                tgt_src_emb_share_weights=True):
        '''
        src_vocab_size, tgt_vocab_size: vocab size of source and target corpus
        len_max_seq: max length of sequences for padding
        d_model: model dim = embedding dim
        inner_dim: dim of inner layer of posisitonal feed forward network
        n_layers: number of layers of encoder, decoder stack
        n_heads: number of head of multihead attention
        d_keys: dim of keys tensor in 1 head (keys and query should have same dim)
        d_values: dim of values tensor in 1 head
        dropout: dropout rate to be used through out the models
        tgt_pred_emb_share_weights: share weights between decoder embedding and fully connected layer
            of decoder
        tgt_src_emb_share_weights: Shared embedding weights between encoder and decoder
        '''
        
        self.len_max_seq = len_max_seq
        self.d_model = d_model
        self.encoder = Encoder(src_vocab_size, len_max_seq,
                                n_layers, n_heads, d_keys, d_values, d_model,
                                inner_dim, dropout=dropout)
        self.decoder = Decoder(tgt_vocab_size, len_max_seq,
                                n_layers, n_heads, d_keys, d_values, d_model,
                                inner_dim, dropout=dropout)
        self.tgt_word_pred = layers.TimeDistributed(layers.Dense(tgt_vocab_size, use_bias=False))

        # if tgt_pred_emb_share_weights:
        #     self.tgt_word_pred.set_weights(self.decoder.tgt_word_emb.get_weights())
        # if tgt_src_emb_share_weights:
        #     self.encoder.src_word_emb.set_weights(self.decoder.tgt_word_emb.get_weights())

    def get_pos_sequence(self, x):
        ''' Position of words that is not padding'''
        mask = K.cast(K.not_equal(x, Const.PAD), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'))
        return pos * mask

    def compile(self, optimizer='adam', **args):
        def padding_loss(args):
            '''Calculate the log loss from output (prob of each words in vocab) with truth'''
            y_pred, y_true = args
            y_true = tf.cast(y_true, 'int32')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, Const.PAD), 'float32')
            loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            loss = K.mean(loss)
            return loss
        def accuracy(args):
            ''' Calculate accuracy of prediction by compare word-wise output and ground truth'''
            y_pred, y_true = args
            mask = K.cast(K.not_equal(y_true, 0), 'float32')
            pred_to_index = K.cast(K.argmax(y_pred, axis=-1), 'int32')
            correlation = K.cast(K.equal(K.cast(y_true, 'int32'), pred_to_index), 'float32')
            correlation = K.sum(correlation * mask, -1)/K.sum(mask, -1)
            return K.mean(correlation)
        src_seq_input = layers.Input(shape=(None,), dtype='int32') # sequence of word index
        tgt_seq_input = layers.Input(shape=(None,), dtype='int32') # sequenc of word index
        tgt_true = layers.Lambda(lambda x: x[:,1:])(tgt_seq_input) # shifted right
        tgt_seq = layers.Lambda(lambda x: x[:,:-1])(tgt_seq_input) # not input <EOS>

        src_seq = src_seq_input
        src_pos = layers.Lambda(self.get_pos_sequence)(src_seq)
        tgt_pos = layers.Lambda(self.get_pos_sequence)(tgt_seq)

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(src_seq, src_pos, tgt_seq, tgt_pos, enc_output)

        final_result = self.tgt_word_pred(dec_output)

        # Custom loss since the output is not normalized with softmax
        prediction_loss = layers.Lambda(padding_loss)([final_result, tgt_true])
        self.ppl = layers.Lambda(K.exp)(prediction_loss)
        self.accuracy = layers.Lambda(accuracy)([final_result, tgt_true])

        # Training model which use log loss as output
        self.training_model = Model(inputs=[src_seq_input, tgt_seq_input], outputs=prediction_loss)
        self.training_model.add_loss([prediction_loss])
        # Use loss=None but with add_loss function, model still run
        # gradient-descent with the prediction loss
        self.training_model.compile(optimizer=optimizer, loss=None)

        # Add accuracy metrics to training model
        self.training_model.metrics_names.append('ppl')
        self.training_model.metrics_tensors.append(self.ppl)
        self.training_model.metrics_names.append('accuracy')
        self.training_model.metrics_tensors.append(self.accuracy)

        # Predict model used for evaluation
        self.make_infer_models()
    
    def make_infer_models(self):
        '''
        Create encode model and decode model separately to 
        '''
        src_seq_input = layers.Input(shape=(None,), dtype='int32')
        tgt_seq_input = layers.Input(shape=(None,), dtype='int32')

        src_seq_pos = layers.Lambda(self.get_pos_sequence)(src_seq_input)
        tgt_seq_pos = layers.Lambda(self.get_pos_sequence)(tgt_seq_input)

        enc_output, *_ = self.encoder(src_seq_input, src_seq_pos)
        self.encode_model = Model(inputs=[src_seq_input], outputs=enc_output)

        enc_res_input = layers.Input(shape=(None, self.d_model))
        dec_output, *_ = self.decoder(src_seq_input, src_seq_pos, tgt_seq_input, tgt_seq_pos, enc_res_input)
        pred = self.tgt_word_pred(dec_output)
        self.decode_model = Model(inputs=[src_seq_input, tgt_seq_input, enc_res_input], outputs=pred)

    def get_infer_model(self):
        return self.encode_model, self.decode_model

class LRSchedulerPerStep(Callback):
    def __init__(self, d_model, warmup=4000):
        self.basic = d_model**-0.5
        self.warm = warmup**-1.5
        self.step_num = 0
    def on_batch_begin(self, batch, logs=None):
        self.step_num += 1
        lr = self.basic * min(self.step_num**-0.5, self.step_num*self.warm)
        K.set_value(self.model.optimizer.lr, lr)