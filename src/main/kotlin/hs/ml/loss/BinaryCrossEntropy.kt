package hs.ml.loss

import hs.ml.math.Tensor

class BinaryCrossEntropy: Loss {
    override fun compute(yTrue: Tensor, yPred: Tensor): Double {
        require(yTrue.shape.second == 1 && yPred.shape.second == 1) {
            "BinaryCrossEntropy can only be computed for single-output tensors."
        }
        require(yTrue.shape.first == yPred.shape.first) {
            "The number of samples in yTrue and yPred must be the same."
        }

        val n = yTrue.shape.first
        var sumCrossEntropy = 0.0
        for (i in 0 until n) {
            val yT = yTrue[i, 0]
            val yP = yPred[i, 0]
            sumCrossEntropy += - (yT * kotlin.math.ln(yP + 1e-15) + (1 - yT) * kotlin.math.ln(1 - yP + 1e-15))
        }

        return sumCrossEntropy / n
    }

    override fun gradient(yTrue: Tensor, yPred: Tensor): Pair<Tensor, Double> {
        require(yTrue.shape.second == 1 && yPred.shape.second == 1) {
            "BinaryCrossEntropy gradient can only be computed for single-output tensors."
        }
        require(yTrue.shape.first == yPred.shape.first) {
            "The number of samples in yTrue and yPred must be the same."
        }

        val n = yTrue.shape.first
        val grad = Tensor(n, 1)
        var biasGrad = 0.0

        for (i in 0 until n) {
            val yT = yTrue[i, 0]
            val yP = yPred[i, 0]
            grad[i, 0] = - (yT / (yP + 1e-15) - (1 - yT) / (1 - yP + 1e-15)) / n
            biasGrad += (yP - yT)
        }


        return Pair(grad, biasGrad)
    }
}