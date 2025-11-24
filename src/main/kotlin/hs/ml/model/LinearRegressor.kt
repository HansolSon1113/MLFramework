package hs.ml.model

import hs.ml.math.Tensor

class LinearRegressor: Regressor() {
    override var weights: Tensor = Tensor(0, 0)
    override var bias: Double = 0.0

    override fun forward(x: Tensor): Tensor {
        return x * weights + bias
    }

    override fun predict(x: Tensor): Tensor {
        return forward(x)
    }

    override fun toString(): String = "LinearRegressor(weights=$weights, bias=$bias, isTrained=$isTrained)"
}
