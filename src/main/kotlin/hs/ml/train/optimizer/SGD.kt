package hs.ml.train.optimizer

import hs.ml.math.Tensor

class SGD : Optimizer {
    override var lr: Double
        private set

    constructor(lr: Double = 0.01) {
        this.lr = lr
    }

    override fun step(params: Pair<Tensor, Double>, gradients: Pair<Tensor, Double>): Pair<Tensor, Double> {
        val w = Tensor(params.first.row, params.first.col) { i, j ->
            params.first[i][j] - (lr * gradients.first[i][j])
        }
        val b = gradients.second.let { params.second - (lr * it) }

        return Pair(w, b)
    }
}