package hs.ml.model

import hs.ml.data.Tensor

interface Model {
    var weights: Tensor
    var bias: Double
    var scaler: Scaler
    var epoch: Int
//    var isTrained: Boolean
//        get() = epoch > 0

    fun predict(x: Tensor): Tensor

    fun evaluate(x: Tensor, y: Tensor, metric: (Tensor, Tensor) -> Double): Double {
        val yhat = predict(x)
        return metric(y, yhat)
    }
}
