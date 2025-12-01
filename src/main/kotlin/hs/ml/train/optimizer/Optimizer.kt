package hs.ml.train.optimizer

import hs.ml.math.Tensor

interface Optimizer {
    val lr: Double

    fun step(params: Pair<Tensor, Double>, gradients: Pair<Tensor, Double>): Pair<Tensor, Double>
}