package hs.ml.scaler

import hs.ml.math.Tensor


abstract class Scaler {
    protected var isTrained = false

    fun fit(x: Tensor) {
        fit(x, Tensor(0,0), 0, 0.0)
    }

    abstract fun fit(x: Tensor, y: Tensor, epochs: Int, lr: Double)
    abstract fun transform(x: Tensor): Tensor
    abstract fun inverse(x: Tensor): Tensor
}
