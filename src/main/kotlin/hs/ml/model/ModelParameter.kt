package hs.ml.model

import hs.ml.loss.Loss
import hs.ml.metric.Metric
import hs.ml.scaler.Scaler

data class ModelParameter(
    var scaler: Scaler? = null,
    var loss: Loss? = null,
    var metric: Metric? = null
)
