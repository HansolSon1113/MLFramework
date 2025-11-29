package hs.ml.train.optimizer

import hs.ml.math.Tensor

interface Optimizer {
    val lr: Double

    fun step(params: Tensor, gradients: Tensor): Tensor
}