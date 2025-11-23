package hs.ml.model

import hs.ml.data.Tensor

class LinearRegressor: Model {
    override var weights: Tensor = Tensor(0, 0)
    override var bias: Double = 0.0
    override lateinit var scaler: Scaler
    override var epoch: Int = 0
    val isTrained: Boolean
        get() = epoch > 0

    override fun predict(xOriginal: Tensor): Tensor {
        require(xOriginal.col == weights.row) { "입력 데이터의 특성 수와 모델의 가중치 수가 일치하지 않습니다." }
        val x = scaler.predict(xOriginal)

        val yhat = Tensor(x.row, 1) { i, j ->
            var sum = 0.0
            for (k in 0 until x.col) {
                sum += x[i, k] * weights[k, 0]
            }
            sum + bias
        }

        return yhat
    }

    override fun toString(): String = "LinearRegressor(weights=$weights, bias=$bias, isTrained=$isTrained)"
}
