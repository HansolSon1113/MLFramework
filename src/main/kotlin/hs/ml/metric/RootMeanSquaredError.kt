package hs.ml.metric

import hs.ml.math.Tensor
import kotlin.math.sqrt

class RootMeanSquaredError: Metric {
    override fun evaluate(yTrue: Tensor, yPred: Tensor): Double {
        require(yTrue.shape == yPred.shape) { "Shapes of yTrue and yPred must be the same." }

        var sumSquaredError = 0.0
        val n = yTrue.row * yTrue.col

        for (i in 0 until yTrue.row) {
            for (j in 0 until yTrue.col) {
                val error = yTrue[i, j] - yPred[i, j]
                sumSquaredError += error * error
            }
        }

        return sqrt(sumSquaredError / n)
    }
}
