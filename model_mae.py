# current implementation: only support numerical values

from functools import partial


import numpy as np
import pandas as pd
import tensorflow as tf
from .utils import MaskEmbed, get_1d_sincos_pos_embed, ActiveEmbed, Block, FeatureKindEmbed
from tensorflow.keras.layers import Dense, LayerNormalization


eps = tf.keras.backend.epsilon()

class MaskedAutoencoder(tf.keras.Model):
    
    """ Masked Autoencoder with Transformer backbone
    """
    
    def __init__(self, rec_len=25, n_value_embed_dim=64, depth=4, num_heads=4,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=LayerNormalization, encode_func='linear',
        mask_ratio=0.2, kind_embed_dim=4,
        ):
        super(MaskedAutoencoder, self).__init__()

        self.mask_ratio = mask_ratio

        # loss tracker for arbitrary custom loss
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        if kind_embed_dim > 0:
            embed_dim = n_value_embed_dim + kind_embed_dim
        else:
            embed_dim = n_value_embed_dim
        
        if encode_func == 'active':
            self.mask_embed = ActiveEmbed(rec_len, embed_dim)
        elif kind_embed_dim > 0:
            self.mask_embed = FeatureKindEmbed(rec_len, n_value_embed_dim, kind_embed_dim=4,)
        else:
            self.mask_embed = MaskEmbed(rec_len, embed_dim)
        
        self.cls_token = tf.Variable(tf.zeros((1, 1, embed_dim)))
        self.pos_embed = tf.Variable(tf.zeros((1, rec_len + 1, embed_dim)), trainable=False)

        self.blocks = [
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, 
            init_values=1e-6,)
            for i in range(depth)
        ]
        self.norm = norm_layer()


        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = Dense(decoder_embed_dim, use_bias=True)

        self.mask_token = tf.Variable(tf.zeros((1, 1, decoder_embed_dim)))
        print(f"self.mask_token.shape {self.mask_token.shape}")

        self.decoder_pos_embed = tf.Variable(tf.zeros((1, rec_len + 1, decoder_embed_dim)), trainable=False)

        self.decoder_blocks = [
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, 
            norm_layer=norm_layer, init_values=1e-6,)
            for i in range(decoder_depth)
        ]

        self.decoder_norm = norm_layer()
        self.decoder_pred = Dense(1, use_bias=True)  # decoder to patch
        
        # --------------------------------------------------------------------------

        self.initialize_weights()


    def initialize_weights(self):
        
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.mask_embed.rec_len, cls_token=True)
        self.pos_embed.assign(tf.convert_to_tensor(pos_embed, dtype=tf.float32)[tf.newaxis])

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.mask_embed.rec_len, cls_token=True)
        self.decoder_pos_embed.assign(tf.convert_to_tensor(decoder_pos_embed, dtype=tf.float32)[tf.newaxis])

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)
        self.cls_token.assign(initializer(shape=self.cls_token.shape))
        self.mask_token.assign(initializer(shape=self.mask_token.shape))


    def random_masking(self, x, flag_notna, mask_ratio, training=True):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        flag_notna: [N, L], 0: nan, 1: not nan,
        mask_ratio: remasking ratio,
        """
        N = tf.shape(x)[0]
        L = tf.shape(x)[1]
        L_float = tf.cast(L, tf.float32)
        D = tf.shape(x)[2]

        # batch, length, dim
        print(N, L, D)

        # 保持する特徴量数を取得
        if training:
            len_keep = int(L_float * (1 - mask_ratio))
        else:
            # TODO: 補完時に欠損が一番多いレコードの利用可能変数の数まで利用変数を減らしてしまう
            len_keep = int(tf.math.reduce_min(tf.reduce_sum(flag_notna, axis=1)))
        print(f"len_keep: {len_keep}")

        # 各バッチで保持・除去する変数を指定するラン数を取得
        noise = tf.random.uniform(shape=(N, L), minval=0, maxval=1, dtype=tf.float32)  # noise in [0, 1]
        # noise = tf.where(m == 0., 1., noise)
        noise_p1 = noise + 1
        noise = tf.where(flag_notna < eps, noise_p1, noise)
        # 乱数の小さい順に特徴量のインデックスを取得
        ids_shuffle = tf.argsort(noise, axis=1)  # ascend: small is keep, large is remove
        # 特徴量ごとに乱数の昇順順位を取得
        ids_restore = tf.argsort(ids_shuffle, axis=1)

        # 保持する特徴量のインデックスを取得
        ids_keep = ids_shuffle[:, :len_keep]
        # 保持する特徴量のみ取得
        x_masked = tf.gather(x, ids_keep, batch_dims=1)

        # マスクを取得: 0 is keep, 1 is remove
        mask = tf.ones([N, L], dtype=tf.float32)

        # バッチごとにlen_keep分だけマスクを0にする
        X, Y = tf.meshgrid(tf.range(N), tf.range(len_keep), indexing='ij')
        indices = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=1)
        updates = tf.zeros([N * len_keep])
        mask = tf.tensor_scatter_nd_update(mask, indices, updates)

        # 乱数の昇順順位がlen_keep以下なら0、それより大きいときは1でマスクする
        mask = tf.gather(mask, ids_restore, batch_dims=1)
        nask = tf.ones([N, L], dtype=tf.float32) - mask

        if training:
            mask = tf.where(flag_notna < eps, 0., mask)
            nask = tf.where(flag_notna < eps, 0., nask)

            return x_masked, mask, nask, ids_restore
        
        else:
            return x_masked, mask, nask, ids_restore



    def forward_encoder(self, x, flag_notna, mask_ratio=0.5, training=None):
        print("\nencoder")

        print(f"x.shape of input: {x.shape}")

        N = tf.shape(x)[0]

        # embed patches (batch_size, n_features, 1) > (batch_size, n_features, embed_dim)
        x = self.mask_embed(x)
        print(f"x.shape after mask_embed {x.shape}")
        print(f"pos_embed.shape {self.pos_embed.shape}")

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x_masked, mask, nask, ids_restore = self.random_masking(x, flag_notna, mask_ratio, training)
        print(f"x.shape after random masking {x_masked.shape}")

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = tf.tile(cls_token, multiples=[N, 1, 1])
        x_masked = tf.concat([cls_tokens, x_masked], axis=1)
        print(f"x.shape after concat cls_tokens {x_masked.shape}")

        # apply Transformer blocks
        for blk in self.blocks:
            x_masked = blk(x_masked)

        x_masked = self.norm(x_masked)

        return x_masked, mask, nask, ids_restore


    def forward_decoder(self, x, ids_restore):
        print("\ndecoder")

        N = tf.shape(x)[0]
        L = tf.shape(x)[1]
        
        # embed tokens
        x = self.decoder_embed(x)

        print(f"self.mask_token.shape {self.mask_token.shape}")
        print(f"ids_restore.shape {ids_restore.shape}")
        # append mask tokens to sequence
        mask_tokens = tf.tile(self.mask_token, multiples=[N, ids_restore.shape[1] + 1 - L, 1])
        print(f"mask_tokens.shape {mask_tokens.shape}")

        x_ = tf.concat([x[:, 1:, :], mask_tokens], axis=1)  # no cls token
        print(f"x_.shape after concating mask tokens {x_.shape}")
        
        x_ = tf.gather(x_, indices=ids_restore, axis=1, batch_dims=1)
        print(f"x_.shape {x_.shape}")

        x = tf.concat([x[:, :1, :], x_], axis=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        # x = tf.math.tanh(self.decoder_pred(x)) / 2.0 + 0.5

        # remove cls token
        x = x[:, 1:, :]

        x = tf.squeeze(x, axis=2)
    
        return x


    def forward_loss(self, x, pred, mask, nask):
        """
        data: [N, L, 1]
        pred: [N, L]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = tf.identity(x)
        
        # loss_ew = tf.math.squared_difference(pred, target)
        loss_ew = (pred - target) ** 2
        loss = tf.reduce_sum(loss_ew * mask) / tf.reduce_sum(mask)
        loss += tf.reduce_sum(loss_ew * nask) / tf.reduce_sum(nask)
        # mean loss on removed patches
        return loss


    def train_step(self, x_and_flag,):

        x, flag_notna = x_and_flag

        with tf.GradientTape() as tape:
            pred, mask, nask = self(x_and_flag, training=True)  # Forward pass
            print(f"pred.shape in train_step: {pred.shape}")
            loss = self.forward_loss(x, pred, mask, nask)
            print(f"loss in train_step: {loss}")

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        print(f"gradients: {gradients}")

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Compute our own metrics
        self.loss_tracker.update_state(loss)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker,]


    def call(self, x_and_flag, training=None):
        x, flag_notna = x_and_flag
        reshaped_x = tf.expand_dims(x, axis=-1)
        
        latent, mask, nask, ids_restore = self.forward_encoder(reshaped_x, flag_notna, self.mask_ratio, training=True)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask, nask
