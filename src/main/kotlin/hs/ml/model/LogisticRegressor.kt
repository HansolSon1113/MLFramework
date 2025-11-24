package hs.ml.model

import hs.ml.math.Tensor
import hs.ml.scaler.Scaler
import kotlin.math.exp

class LogisticRegressor : Model {
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
            var z = 0.0
            for (k in 0 until x.col) {
                z += x[i, k] * weights[k, 0]
            }
            z += bias
            1.0 / (1.0 + exp(-z))
        }

        return yhat
    }

    fun classify(xOrg: Tensor, threshold: Double = 0.5): Tensor {
        val prob = predict(xOrg)
        return Tensor(prob.row, 1) { i, _ ->
            if (prob[i, 0] >= threshold) 1.0 else 0.0
        }
    }

    override fun toString(): String = "LinearRegressor(weights=$weights, bias=$bias, isTrained=$isTrained)"
}