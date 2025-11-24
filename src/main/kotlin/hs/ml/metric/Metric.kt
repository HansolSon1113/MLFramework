package hs.ml.metric

import hs.ml.math.Tensor

@FunctionalInterface
interface Metric {
    fun evaluate(yTrue: Tensor, yPred: Tensor): Double
}
