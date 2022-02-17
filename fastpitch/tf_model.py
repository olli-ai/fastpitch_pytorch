import pdb
from typing import Optional
import tensorflow as tf
import numpy as np


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = tf.range(0, max_len, dtype=lens.dtype)
    mask = tf.math.less(ids, tf.expand_dims(lens, 1)) # !note: origin torch function is torch.lt (less_than)
    return mask

def regulate_length(durations, enc_out, pace: float=1.0,
                    mel_max_len: Optional[int] = None):
    """if target = None then the predicted durations are applied

    Args:
        durations ([batch_size, token_len]): duration of each token in enc_out
        enc_out ([bacth_size, token_len, ]): output of encoder (read more in paper)
        pace (float, optional): control the speed of speech. Defaults to 1.0.
        mel_max_len (Optional[int], optional): Max len of speech. Defaults to None.

    Returns:
        [type]: [description]
    """
    dtype = enc_out.dtype # [BS, T, H]
    print(durations)
    reps = tf.cast(durations, dtype=tf.float32)/pace
    reps = tf.cast((reps + 0.5), dtype=tf.int32)
    dec_lens = tf.reduce_sum(reps, 1, keepdims=True)
    max_len = tf.reduce_max(dec_lens)
    
    reps_cumsum = tf.cumsum(
        tf.pad(reps, tf.constant([[0,0], [1, 0]]), constant_values = 0.0), # !NOTE: the order of padding tensor is different between torch and tf
        axis=1
    )
    
    reps_cumsum = tf.expand_dims(reps_cumsum, 1) # Equal to [:, None, :] in torch
    reps_cumsum = tf.cast(reps_cumsum, dtype) # [BS, 1, T+1]
    
    range_ = tf.range(max_len)
    range_ = tf.expand_dims(tf.expand_dims(range_, -1), 0)
    range_ = tf.cast(range_, tf.float32)
    mult = ((reps_cumsum[:, :, :-1] <= range_) &
            (reps_cumsum[:, :, 1:] > range_))
    print("Mult", mult)
    mult = tf.cast(mult, dtype)
    enc_rep = tf.matmul(mult, enc_out)
    
    if mel_max_len is not None:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = tf.clip_by_value(dec_lens, -1, mel_max_len)
    return enc_rep, dec_lens

def average_pitch(pitch, durs):
    """ Local resolution of pitch per token (read more on paper)

    Args:
        pitch ([batch_size x 1 x p]): predicted/groundtruth pitch with len = p
        durs ([batch_size x d]): predicted/groundtruth duration with len = d 
    """
    
    durs_cums_ends = tf.cast(tf.cumsum(durs, axis=1), tf.int32) # [batch_size * d]
    durs_cums_starts = tf.pad(durs_cums_ends[:, :-1], tf.constant([[0, 0], [1, 0]]))
    pitch_nonzero_cums = tf.pad(tf.cumcum(pitch != 0.0, axis=2), tf.constant([[0, 0], [0, 0],[1, 0]]))
    pitch_cums = tf.pad(tf.cumsum(pitch, axis=2), tf.constant([[0, 0], [0, 0], [1, 0]]))

    bs, l = durs_cums_ends.shape
    n_formants = pitch.shape[1]

    dcs = tf.expand_dims(durs_cums_starts, 1) # [batch_size * 1 * d]
    dce = tf.expand_dims(durs_cums_ends, 1)

    dcs = tf.tile(dcs, [1, n_formants, 1]) # [bacth_size * n_formant * d]
    dce = tf.tile(dce, [1, n_formants, 1])

    pitch_sums = tf.cast((tf.gather(pitch_cums, dce, axis=2) [0, 0]
                    - tf.gather(pitch_cums, dcs, axis=2))[0, 0], tf.float32)
    pitch_nelems = tf.cast((tf.gather(pitch_nonzero_cums, dce, axis=2)[0, 0]
                    - tf.gather(pitch_nonzero_cums, dcs, axis=2)[0, 0]), tf.float32)
    pitch_avg = tf.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums/pitch_nelems)

    return pitch_avg

class ConvReLUNorm(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super(ConvReLUNorm, self).__init__()
        self.conv = tf.keras.layers.Conv1D(
            filters=out_channels,
            kernel_size=kernel_size,
            padding='same', # (kernel_size // 2)
            activation=tf.keras.activations.relu,
        )
        
        self.norm = tf.keras.layers.LayerNormalization(
            axis=1, epsilon=1e-05
        )
        
        self.dropout = tf.keras.layers.Dropout(
            rate=dropout
        )
        
    def call(self, signal, training=False):
        out = self.conv(signal)
        out = self.norm(out)
        return self.dropout(out)
    

class TemporalPredictor(tf.keras.layers.Layer):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout,
                 n_layers=2, n_predictions=1):
        super(TemporalPredictor, self).__init__()

        self.layers = tf.keras.Sequential([
            ConvReLUNorm(input_size if i == 0 else filter_size, filter_size, kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)])

        self.fc = tf.keras.layers.Dense(n_predictions, use_bias=True)

    def __call__(self, enc_out, enc_out_mask):
        # TODO: mismatch shape between matrices
        out = enc_out * tf.cast(enc_out_mask, tf.float32) # [Batch_size, tokens_len, input_hidden_features_len]
        out = self.layers(out) # [BS, T, H]
        out = self.fc(out) # 
        out = out * tf.cast(enc_out_mask, tf.float32) # [Batch_size, tokens_len, 1]
        return out

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        self.inv_freq = 1 / (10000 ** (tf.range(0.0, demb, 2.0, tf.float32) / demb))
        # self.register_buffer('inv_freq', inv_freq)

    def call(self, pos_seq, bsz=None):
        sinusoid_inp = tf.matmul(tf.expand_dims(pos_seq, -1),
                                    tf.expand_dims(self.inv_freq, 0))
        pos_emb = tf.concat([tf.math.sin(sinusoid_inp), tf.math.cos(sinusoid_inp)], axis=1)
        if bsz is not None:
            return tf.tile(tf.expand_dims(pos_emb, 0), [bsz, 1, 1])
        else:
            return tf.expand_dims(pos_emb, 0)

class MultiHeadAttn(tf.keras.layers.Layer):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.1,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.qkv_net = tf.keras.layers.Dense(3 * n_head * d_head)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.dropatt = tf.keras.layers.Dropout(dropatt)
        self.o_net = tf.keras.layers.Dense(d_model, use_bias=False)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, inp, attn_mask=None):
        """Call logic"""
        return self._forward(inp, attn_mask)

    def _forward(self, inp, attn_mask=None):
        residual = inp

        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp)

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = tf.split(self.qkv_net(inp), num_or_size_splits=3, axis=2)
        head_q = tf.reshape(head_q, [inp.shape[0], inp.shape[1], n_head, d_head])
        head_k = tf.reshape(head_k, [inp.shape[0], inp.shape[1], n_head, d_head])
        head_v = tf.reshape(head_v, [inp.shape[0], inp.shape[1], n_head, d_head])

        q = tf.reshape(tf.transpose(head_q, perm=[0, 2, 1, 3]), [-1, inp.shape[1], d_head])
        k = tf.reshape(tf.transpose(head_k, perm=[0, 2, 1, 3]), [-1, inp.shape[1], d_head])
        v = tf.reshape(tf.transpose(head_v, perm=[0, 2, 1, 3]), [-1, inp.shape[1], d_head])

        attn_score = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
        attn_score = attn_score * self.scale

        if attn_mask is not None:
            attn_mask = tf.cast(tf.expand_dims(attn_mask, 1), attn_score.dtype)
            attn_mask = tf.tile(attn_mask, [n_head, attn_mask.shape[2], 1])
            # TODO: reimplement the attention mask
            attn_mask = tf.where(tf.cast(attn_mask, tf.bool), attn_mask, tf.fill(attn_mask.shape, float('-inf')))
            attn_score = attn_score + attn_mask
            # !note: this line maybe not work: attn_score.masked_fill_(attn_mask.to(tf.bool), -float('inf'))

        attn_prob = tf.nn.softmax(attn_score, axis=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = tf.matmul(attn_prob, v)

        attn_vec = tf.reshape(attn_vec, [n_head, inp.shape[0], inp.shape[1], d_head])
        attn_vec = tf.reshape(tf.transpose(attn_vec, [1,2,0,3]), [inp.shape[0], inp.shape[1], n_head * d_head])

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out)

        output = tf.cast(output, attn_out.dtype)

        return output

class PositionwiseConvFF(tf.keras.layers.Layer):
    def __init__(self, d_model, d_inner, kernel_size, dropout, pre_lnorm=False):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        # print("tf_model:", kernel_size)

        self.CoreNet = tf.keras.Sequential(
            [tf.keras.layers.Conv1D(d_inner, kernel_size, 1, 'same', activation='relu'), # !note: the padding may not work because it's was implicitly define in torch instead of mode-only
            # tf.nn.ReLU(),
            # nn.Dropout(dropout),  # worse convergence
            tf.keras.layers.Conv1D(d_model, kernel_size, 1, 'same'),
            tf.keras.layers.Dropout(rate=dropout)]
            # nn.Dropout(dropout),
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.pre_lnorm = pre_lnorm

    def call(self, inp):
        """Call logic"""
        return self._forward(inp)

    def _forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = tf.transpose(inp, perm=[0, 2, 1])
            core_out = self.CoreNet(tf.cast(self.layer_norm(core_out), inp.dtype))
            core_out = tf.transpose(core_out, perm=[0, 2, 1])

            # residual connection
            output = core_out + inp
        else:
            # # positionwise feed-forward
            # import pdb; pdb.set_trace()
            # core_out = tf.transpose(inp, perm=[0, 2, 1])
            core_out = inp
            core_out = self.CoreNet(core_out)
            # core_out = tf.transpose(core_out, perm=[0, 2, 1])

            # residual connection + layer normalization
            output = tf.cast(self.layer_norm(inp + core_out), inp.dtype)
            # print(output.shape)

        return output

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, n_head, d_model, d_head, d_inner, kernel_size, dropout,
                 **kwargs):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseConvFF(d_model, d_inner, kernel_size, dropout,
                                         pre_lnorm=kwargs.get('pre_lnorm'))

    def call(self, dec_inp, mask=None):
        # print("tf_model:", dec_inp)
        output = self.dec_attn(dec_inp, attn_mask=~tf.squeeze(mask, 2))
        output = tf.math.multiply(output, tf.cast(mask, tf.float32))
        output = self.pos_ff(output)
        # print("tf_model:", output)
        output = tf.math.multiply(output, tf.cast(mask, tf.float32))
        return output

class FFTransformer(tf.keras.layers.Layer):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, kernel_size,
                 dropout, dropatt, dropemb=0.0, embed_input=True,
                 n_embed=None, d_embed=None, padding_idx=0, pre_lnorm=False):
        super(FFTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.padding_idx = padding_idx

        if embed_input:
            self.word_emb = tf.keras.layers.Embedding(n_embed, d_embed or d_model) # !note: padding_idx is used in torch but no coressponding param
        else:
            self.word_emb = None

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = tf.keras.layers.Dropout(dropemb)
        self.layers =  [TransformerLayer(
                                n_head, d_model, d_head, d_inner, kernel_size, dropout,
                                dropatt=dropatt, pre_lnorm=pre_lnorm)
                        for _ in range(n_layer)]

    def call(self, dec_inp, seq_lens=None, conditioning=0):
        if self.word_emb is None:
            inp = dec_inp
            mask = tf.expand_dims(mask_from_lens(seq_lens), 2)
        else:
            inp = self.word_emb(dec_inp)
            # [bsz x L x 1]
            mask = tf.expand_dims(dec_inp != self.padding_idx, 2)

        pos_seq = tf.range(inp.shape[1], dtype=inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * tf.cast(mask, tf.float32)

        out = self.drop(inp + pos_emb + conditioning)

        for layer in self.layers:
            out = layer(out, mask=mask)

        # out = self.drop(out)
        return out, mask

class ConvAttention(tf.keras.layers.Layer):
    ...

class FastPitch(tf.keras.Model):
    def __init__(self, n_mel_channels, n_symbols, padding_idx,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size,
                 p_in_fft_dropout, p_in_fft_dropatt, p_in_fft_dropemb,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 out_fft_output_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 dur_predictor_kernel_size, dur_predictor_filter_size,
                 p_dur_predictor_dropout, dur_predictor_n_layers,
                 pitch_predictor_kernel_size, pitch_predictor_filter_size,
                 p_pitch_predictor_dropout, pitch_predictor_n_layers,
                 pitch_embedding_kernel_size,
                 energy_conditioning,
                 energy_predictor_kernel_size, energy_predictor_filter_size,
                 p_energy_predictor_dropout, energy_predictor_n_layers,
                 energy_embedding_kernel_size,
                 n_speakers, speaker_emb_weight, pitch_conditioning_formants=1):
        super(FastPitch, self).__init__()

        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers, n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=True,
            d_embed=symbols_embedding_dim,
            n_embed=n_symbols,
            padding_idx=padding_idx)

        if n_speakers > 1:
            self.speaker_emb = tf.keras.layers.Embedding(n_speakers, symbols_embedding_dim)
        else:
            self.speaker_emb = None
        self.speaker_emb_weight = speaker_emb_weight

        self.duration_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=dur_predictor_filter_size,
            kernel_size=dur_predictor_kernel_size,
            dropout=p_dur_predictor_dropout, n_layers=dur_predictor_n_layers
        )

        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )

        self.pitch_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=pitch_predictor_filter_size,
            kernel_size=pitch_predictor_kernel_size,
            dropout=p_pitch_predictor_dropout, n_layers=pitch_predictor_n_layers,
            n_predictions=pitch_conditioning_formants
        )
        self.pitch_emb = tf.keras.layers.Conv1D(
            filters=symbols_embedding_dim,
            kernel_size=pitch_embedding_kernel_size,
            padding='same'
        )

        # Store values precomputed for training data within the model
        self.pitch_mean = tf.Variable(tf.constant([0.], tf.float32))
        self.pitch_std = tf.Variable(tf.constant([0.], tf.float32))

        self.energy_conditioning = energy_conditioning
        if energy_conditioning:
            self.energy_predictor = TemporalPredictor(
                in_fft_output_size,
                filter_size=energy_predictor_filter_size,
                kernel_size=energy_predictor_kernel_size,
                dropout=p_energy_predictor_dropout,
                n_layers=energy_predictor_n_layers,
                n_predictions=1
            )

            self.energy_emb = tf.keras.layers.Conv1D(
                symbols_embedding_dim,
                kernel_size=energy_embedding_kernel_size,
                padding='same')

        # self.proj = nn.Linear(out_fft_output_size, n_mel_channels, bias=True)
        self.proj = tf.keras.layers.Dense(n_mel_channels, use_bias=True)

        # self.attention = ConvAttention(
        #     n_mel_channels, 0, symbols_embedding_dim,
        #     use_query_proj=True, align_query_enc_type='3xconv')

    def _build(self):
        """Dummy input for building model."""
        # inputs
        # input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        input_ids = tf.cast(tf.expand_dims(tf.range(1,36), 0), tf.int32)
        self(
            inputs=input_ids
        )

    def call(self, inputs, pace=1.0, dur_tgt=None, pitch_tgt=None,
              energy_tgt=None, pitch_transform=None, max_duration=75,
              speaker=0):
        print("+"*50)
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            speaker = tf.ones(inputs.shape[0], tf.float32) * speaker
            spk_emb = tf.expand_dims(self.speaker_emb(speaker), 1)
            # print(spk_emb.shape, self.speaker_emb_weight.shape)
            spk_emb = tf.math.multiply(spk_emb, self.speaker_emb_weight)

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        # Predict durations
        log_dur_pred = tf.squeeze(self.duration_predictor(enc_out, enc_mask), axis=-1)
        # print("Log duration pred: ", log_dur_pred)
        dur_pred = tf.clip_by_value(tf.math.exp(log_dur_pred) - 1, 0, max_duration)

        # Pitch over chars
        pitch_pred = self.pitch_predictor(enc_out, enc_mask)
        print("Pitch pred: ", pitch_pred)
        # pitch_pred = tf.transpose(pitch_pred, perm=[0,2,1])

        if pitch_transform is not None:
            if self.pitch_std[0] == 0.0:
                # XXX LJSpeech-1.1 defaults
                mean, std = 218.14, 67.24
            else:
                mean, std = self.pitch_mean[0], self.pitch_std[0]
            pitch_pred = pitch_transform(pitch_pred, enc_mask.sum(dim=(1,2)),
                                         mean, std)
        if pitch_tgt is None:
            pitch_emb = self.pitch_emb(pitch_pred)
            # pitch_emb = tf.transpose(pitch_emb, perm=[0,2,1])
        else:inputs
        # Predict energy
        if self.energy_conditioning:

            if energy_tgt is None:
                energy_pred = self.energy_predictor(enc_out, enc_mask)
                energy_pred = tf.squeeze(energy_pred, -1)
                energy_pred = tf.expand_dims(energy_pred, 2)
                energy_emb = self.energy_emb(energy_pred)
                # energy_emb = tf.transpose(energy_emb, perm=[0,2,1])

            else:
                energy_emb = self.energy_emb(energy_tgt)
                energy_emb = tf.transpose(energy_emb, perm=[0,2,1])
            print(f"{enc_out.shape}  + {energy_emb.shape}")
            enc_out = enc_out + energy_emb
        else:
            energy_pred = None

        
        len_regulated, dec_lens = regulate_length(
            dur_pred if dur_tgt is None else dur_tgt,
            enc_out, pace, mel_max_len=None)

        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        # mel_lens = dec_mask.squeeze(2).sum(axis=1).long()
        mel_out = mel_out.permute(0, 2, 1)  # For inference.py
        return mel_out, dec_lens, dur_pred, pitch_pred, energy_pred