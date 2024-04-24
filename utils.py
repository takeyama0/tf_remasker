# current implementation: only support numerical values
import numpy as np
import os
import pandas as pd
import math
import argparse
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, MultiHeadAttention, LayerNormalization, Identity
import tensorflow_models as tfm
from functools import partial


class Block(tf.keras.layers.Layer):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=tf.keras.layers.Activation(tf.keras.activations.gelu),
            norm_layer=LayerNormalization
    ):
        super().__init__()
        self.norm1 = norm_layer()
        self.attn = tfm.nlp.layers.TalkingHeadsAttention(num_heads=num_heads, key_dim=dim, value_dim=dim, use_bias=qkv_bias, dropout=attn_drop)
        self.ls1 = tfm.vision.layers.Scale(initializer=tf.keras.initializers.Constant(value=init_values)) if init_values else Identity()
        self.drop_path1 = tfm.vision.layers.StochasticDepth (drop_path) if drop_path > 0. else Identity()

        self.norm2 = norm_layer()
        self.mlp = MLP(units_hidden=int(dim * mlp_ratio), units_out=dim, act_layer=act_layer, drop=drop)
        self.ls2 = tfm.vision.layers.Scale(initializer=tf.keras.initializers.Constant(value=init_values)) if init_values else Identity()
        self.drop_path2 = tfm.vision.layers.StochasticDepth (drop_path) if drop_path > 0. else Identity()


    def call(self, x):
        normed_x = self.norm1(x)
        attention_output = self.attn(normed_x, normed_x)
        x = x + self.drop_path1(self.ls1(attention_output))
        normed_x = self.norm2(x)
        x = x + self.drop_path2(self.ls2(self.mlp(normed_x)))
        return x


class MLP(tf.keras.layers.Layer):
    def __init__(
        self,
        units_hidden,
        units_out,
        act_layer=tf.keras.layers.Activation(tf.keras.activations.gelu),
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super(MLP, self).__init__()
        self.units_out = units_out
        self.units_hidden = units_hidden
        linear_layer = partial(tf.keras.layers.Conv2D, kernel_size=1,) if use_conv else tf.keras.layers.Dense

        self.fc1 = linear_layer(units_hidden, use_bias=bias)
        self.act = act_layer
        self.drop1 = tf.keras.layers.Dropout(drop)
        self.fc2 = linear_layer(units_out, use_bias=bias)
        self.drop2 = tf.keras.layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MaskEmbed(tf.keras.layers.Layer):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, norm_layer=None):
        super(MaskEmbed, self).__init__()
        self.rec_len = rec_len
        self.proj = Conv1D(embed_dim, kernel_size=1, strides=1, data_format="channels_last")
        self.norm = norm_layer() if norm_layer else tf.identity

    def call(self, x):
        x = self.proj(x)
        # TODO:Transposeいらなくない？
        # x = tf.transpose(x, perm=[0, 2, 1])
        x = self.norm(x)
        return x


class ActiveEmbed(tf.keras.layers.Layer):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, norm_layer=None):
        super(MaskEmbed, self).__init__()
        self.rec_len = rec_len
        self.proj = Conv1D(embed_dim, kernel_size=1, strides=1, data_format="channels_last")
        self.norm = norm_layer() if norm_layer else tf.identity

    def call(self, x):
        x = self.proj(x)
        x = tf.math.sin(x)
        # TODO:Transposeいらなくない？
        # x = tf.transpose(x, perm=[0, 2, 1])
        x = self.norm(x)
        return x


class FeatureKindEmbed(tf.keras.layers.Layer):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, kind_embed_dim=8, norm_layer=None):
        super(FeatureKindEmbed, self).__init__()
        self.rec_len = rec_len
        self.feature_kind_embed = tf.keras.layers.Embedding(input_dim=rec_len, output_dim=kind_embed_dim,)
        self.proj = Conv1D(embed_dim, kernel_size=1, strides=1, data_format="channels_last")
        self.norm = norm_layer() if norm_layer else tf.identity

    def call(self, x):
        
        N = tf.shape(x)[0]
        N_float = tf.cast(N, tf.float32)

        feature_kinds = tf.expand_dims(tf.range(self.rec_len), axis=0)
        feature_kinds = tf.tile(feature_kinds, multiples=[N, 1])
        feature_kinds = self.feature_kind_embed(feature_kinds)

        x = self.proj(x)
        x = self.norm(x)
        x = tf.concat([x, feature_kinds], axis=-1)
        return x


class FeatureKindDense(tf.keras.layers.Layer):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, kind_embed_dim=8, norm_layer=None):
        super(FeatureKindDense, self).__init__()
        self.rec_len = rec_len
        self.feature_kind_embed = tf.keras.layers.Embedding(input_dim=rec_len, output_dim=kind_embed_dim,)
        self.proj = Dense(embed_dim)
        self.norm = norm_layer() if norm_layer else tf.identity

    def call(self, x):
        
        N = tf.shape(x)[0]
        N_float = tf.cast(N, tf.float32)

        feature_kinds = tf.range(self.rec_len)
        feature_kinds = tf.tile(feature_kinds, multiples=[N, 1])
        feature_kinds = self.feature_kind_embed(feature_kinds)

        x = tf.concat([x, feature_kinds], axis=-1)
        x = self.proj(x)
        x = self.norm(x)
        return x


def get_1d_sincos_pos_embed(embed_dim, n_pos, cls_token=False):
    """
    embed_dim: output dimension for each position
    n_pos: number of positions to be encoded
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(n_pos)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed
