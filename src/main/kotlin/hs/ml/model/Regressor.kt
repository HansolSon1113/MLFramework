package hs.ml.model

import hs.ml.math.Tensor

abstract class Regressor: Model() {
    abstract fun predict(x: Tensor): Tensor
}
