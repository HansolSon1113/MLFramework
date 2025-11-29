package hs.ml.train.optimizer

import hs.ml.math.Tensor

class SGD : Optimizer {
    override var lr: Double
        private set

    constructor(lr: Double = 0.01) {
        this.lr = lr
    }

    override fun step(params: Tensor, gradients: Tensor): Tensor {
        return Tensor(params.row, params.col) { i, j ->
            params[i][j] - (lr * gradients[i][j])
        }
    }
}