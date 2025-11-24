package hs.ml.model

import hs.ml.math.Tensor

abstract class Model {
    protected abstract var weights: Tensor
    protected abstract var bias: Double
    protected abstract var isTrained: Boolean
    protected abstract var param: ModelParameter

    abstract fun forward(x: Tensor): Tensor
}
