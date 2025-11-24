package hs.ml.scaler

import hs.ml.math.Tensor
import hs.ml.model.Model


interface Scaler: Model {
    fun fit(x: Tensor) {
        fit(x, Tensor(0,0), 0, 0.0)
    }

    fun fit(x: Tensor, y: Tensor, epochs: Int, lr: Double)

    fun inverseTransform(x: Tensor): Tensor
}
