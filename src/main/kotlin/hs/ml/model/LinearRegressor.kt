package hs.ml.model

import hs.ml.data.Tensor

class LinearRegressor: Model {
    var weights: Tensor = Tensor(0, 0)
    var bias: Double = 0.0
    var isTrained: Boolean = false
    lateinit var scaler: StandardScaler

    override fun fit(xOriginal: Tensor, y: Tensor, epochs: Int, lr: Double) {
        scaler = StandardScaler()
        scaler.fit(xOriginal, y, epochs, lr)
        val x = scaler.predict(xOriginal)

        weights = Tensor(x.col, 1) { i, j -> 0.0 }
        bias = 0.0

        for (epoch in 0 until epochs) {
            val yhat = predict(x)
            val errors = Tensor(y.row, 1) { i, j -> yhat[i, j] - y[i, j] }

            println("Epoch ${epoch + 1}/$epochs - Loss: ${Evaluator.mse(y, yhat)}")

            // 가중치와 편향 업데이트
            for (j in 0 until x.col) {
                var gradient = 0.0
                for (i in 0 until x.row) {
                    gradient += errors[i, 0] * x[i, j] / x.row
                }
                weights[j, 0] -= lr * gradient
            }

            var biasGradient = 0.0
            for (i in 0 until x.row) {
                biasGradient += errors[i, 0] / x.row
            }
            bias -= lr * biasGradient
        }

        isTrained = true
    }

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
