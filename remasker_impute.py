# stdlib
from typing import Any, List, Tuple, Union

# third party
import numpy as np
import math, sys, argparse
import pandas as pd
import tensorflow as tf
from functools import partial
import time, os, json
from . import model_mae
import sys

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import MissingIndicator
from tensorflow.keras.layers import BatchNormalization, LayerNormalization

eps = 1e-8

class ReMasker:

    def __init__(self, batch_size=64, accum_iter=1, min_lr=1e-5, weight_decay=0.05, lr=None,
                 blr=1e-3, warmup_epochs=40, n_value_embed_dim=32, depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4.0,
                 max_epochs=600, mask_ratio=0.5, encode_func='linear'
                 ):
        self.batch_size = batch_size
        self.accum_iter = accum_iter
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.lr = lr
        self.blr = blr
        self.warmup_epochs = warmup_epochs
        self.model = None
        self.norm_parameters = None
        self.scaler = None
        
        self.n_value_embed_dim = n_value_embed_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.max_epochs = max_epochs
        self.mask_ratio = mask_ratio
        self.encode_func = encode_func

    def fit(self, X_raw: pd.DataFrame):
        X = X_raw.copy()

        # Parameters
        no, dim = X.shape

        # TODO: StandardScalerとMinMaxScaler選べるようにする
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Set missing
        flag_notna = 1. - (1. * (np.isnan(X)))
        X = np.where(np.isnan(X), np.zeros_like(X), X)

        self.model = model_mae.MaskedAutoencoder(
            rec_len=dim,
            n_value_embed_dim=self.n_value_embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.n_value_embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=BatchNormalization,
            encode_func=self.encode_func,
            mask_ratio=self.mask_ratio
        )

        self.model.compile(optimizer='adam')
        self.hisotry = self.model.fit(X, flag_notna, epochs=self.max_epochs)

        return self

        
    def transform(self, X_raw: pd.DataFrame):

        X = X_raw.copy()
        no, dim = X.shape

        # normalization
        X = self.scaler.transform(X)

        # Set missing
        flag_notna = 1. - (1. * (np.isnan(X)))
        X = np.where(np.isnan(X), np.zeros_like(X), X)

        # Impute missing
        imputed_data = self.model.predict((X, flag_notna))[0]
        imputed_data = self.scaler.inverse_transform(imputed_data)
        imputed_data = np.where(np.isnan(X_raw), imputed_data, X_raw,)

        if np.any(np.isnan(imputed_data)):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)

        return imputed_data

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Imputes the provided dataset using the GAIN strategy.
        Args:
            X: np.ndarray
                A dataset with missing values.
        Returns:
            Xhat: The imputed dataset.
        """
        columns = X.columns
        return pd.DataFrame(self.fit(X).transform(X), columns=columns)

