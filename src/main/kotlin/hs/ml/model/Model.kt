package hs.ml.model

import hs.ml.math.Tensor

abstract class Model {
    abstract var weights: Tensor
    abstract var bias: Double
    var param: ModelParameter = ModelParameter()
    var isTrained: Boolean = false

    abstract fun forward(x: Tensor): Tensor
}
