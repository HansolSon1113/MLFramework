package hs.ml.model

import hs.ml.math.Tensor
import kotlin.math.exp

class LogisticRegressor : Classifier() {
    override var weights: Tensor = Tensor(0, 0)
    override var bias: Double = 0.0

    override fun forward(x: Tensor): Tensor {
        require(x.col == weights.row) { "입력 데이터의 특성 수와 모델의 가중치 수가 일치하지 않습니다." }

        val yhat = Tensor(x.row, 1) { i, j ->
            var linearSum = 0.0
            for (k in 0 until x.col) {
                linearSum += x[i, k] * weights[k, 0]
            }
            val z = linearSum + bias
            1 / (1 + exp(-z))
        }

        return yhat
    }

    override fun classify(x: Tensor, threshold: Double): Tensor {
        val probabilities = forward(x)
        val predictions = Tensor(probabilities.row, 1) { i, j ->
            if (probabilities[i, 0] >= threshold) 1.0 else 0.0
        }
        return predictions
    }

    override fun toString(): String = "LinearRegressor(weights=$weights, bias=$bias, isTrained=$isTrained)"
}