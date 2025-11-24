package hs.ml.model

import hs.ml.scaler.Scaler

data class ModelParameter(
    var scaler: Scaler? = null,
)
