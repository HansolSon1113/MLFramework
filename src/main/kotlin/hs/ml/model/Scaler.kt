package hs.ml.model

import hs.ml.data.Tensor


interface Scaler: Model {
    fun fit(x: Tensor) {
        fit(x, Tensor(0,0), 0, 0.0)
    }

    fun fit(x: Tensor, y: Tensor, epochs: Int, lr: Double)

    fun inverseTransform(x: Tensor): Tensor
}
