package hs.ml.loss

import hs.ml.math.Tensor

class MeanSquaredError: Loss {
    override fun compute(yTrue: Tensor, yPred: Tensor): Double {
        require(yTrue.shape.second == 1 && yPred.shape.second == 1) {
            "MeanSquaredError can only be computed for single-output tensors."
        }
        require(yTrue.shape.first == yPred.shape.first) {
            "The number of samples in yTrue and yPred must be the same."
        }

        val n = yTrue.shape.first
        var sumSquaredError = 0.0

        for (i in 0 until n) {
            val error = yTrue[i, 0] - yPred[i, 0]
            sumSquaredError += error * error
        }

        return sumSquaredError / n
    }

    override fun gradient(yTrue: Tensor, yPred: Tensor): Pair<Tensor, Double> {
        require(yTrue.shape.second == 1 && yPred.shape.second == 1) {
            "MeanSquaredError gradient can only be computed for single-output tensors."
        }
        require(yTrue.shape.first == yPred.shape.first) {
            "The number of samples in yTrue and yPred must be the same."
        }

        val n = yTrue.shape.first
        val grad = Tensor(n, 1)
        var biasGrad = 0.0

        for (i in 0 until n) {
            grad[i, 0] = -2.0 * (yTrue[i, 0] - yPred[i, 0]) / n
            biasGrad += (yPred[i, 0] - yTrue[i, 0]) * 2.0 / n
        }

        return Pair(grad, biasGrad)
    }
}
