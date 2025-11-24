package hs.ml.metric

import hs.ml.math.Tensor

class F1: Metric {
    override fun evaluate(yTrue: Tensor, yPred: Tensor): Double {
        require(yTrue.shape.second == 1) { "yTrue and yPred must be column matrix" }
        require(yTrue.shape == yPred.shape) { "Shapes of yTrue and yPred must be the same." }

        var tp = 0.0
        var fp = 0.0
        var fn = 0.0

        for (i in 0 until yTrue.row) {
            val trueLabel = yTrue[i, 0]
            val predLabel = yPred[i, 0]

            if (trueLabel == 1.0 && predLabel == 1.0) {
                tp += 1.0
            } else if (trueLabel == 0.0 && predLabel == 1.0) {
                fp += 1.0
            } else if (trueLabel == 1.0 && predLabel == 0.0) {
                fn += 1.0
            }
        }

        val precision = if (tp + fp == 0.0) 0.0 else tp / (tp + fp)
        val recall = if (tp + fn == 0.0) 0.0 else tp / (tp + fn)

        return if (precision + recall == 0.0) 0.0 else 2 * (precision * recall) / (precision + recall)
    }
}