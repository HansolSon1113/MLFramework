package hs.ml.loss

import hs.ml.math.Tensor

interface Loss {
    fun compute(yTrue: Tensor, yPred: Tensor): Double
    fun gradient(yTrue: Tensor, yPred: Tensor): Pair<Tensor, Double>
}
