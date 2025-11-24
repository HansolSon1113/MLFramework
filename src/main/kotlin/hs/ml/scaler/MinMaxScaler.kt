package hs.ml.scaler

import hs.ml.math.Tensor

class MinMaxScaler: Scaler() {
    private var min: Double = 0.0
    private var max: Double = 0.0

    override fun fit(x: Tensor, y: Tensor, epochs: Int, lr: Double) {
        min = x.min()
        max = x.max()
        isTrained = true
    }

    override fun transform(x: Tensor): Tensor {
        return Tensor(x.row, x.col) { i, j ->
            (x[i, j] - min) / (max - min)
        }
    }

    override fun inverse(x: Tensor): Tensor {
        return Tensor(x.row, x.col) { i, j ->
            x[i, j] * (max - min) + min
        }
    }
}
