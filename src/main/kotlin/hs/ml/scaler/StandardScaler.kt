package hs.ml.scaler

import hs.ml.math.Tensor
import hs.ml.math.TensorFactory
import kotlin.math.sqrt

class StandardScaler : Scaler() {
    private lateinit var mean: Tensor
    private lateinit var std: Tensor

    override fun fit(x: Tensor, y: Tensor, epochs: Int, lr: Double) {
        val cols = x.col
        mean = TensorFactory.create(1, cols) { j, _ ->
            var sum = 0.0
            for (i in 0 until x.row) {
                sum += x[i, j]
            }
            sum / x.row
        }

        std = TensorFactory.create(1, cols) { j, _ ->
            var sumSq = 0.0
            for (i in 0 until x.row) {
                val diff = x[i, j] - mean[0, j]
                sumSq += diff * diff
            }
            sqrt(sumSq / x.row)
        }

        isTrained = true
    }

    override fun transform(x: Tensor): Tensor {
        require(isTrained) { "Scaler must be fitted before transformation" }

        return TensorFactory.create(x.row, x.col) { i, j ->
            if (std[0, j] != 0.0) {
                (x[i, j] - mean[0, j]) / std[0, j]
            } else {
                0.0
            }
        }
    }

    override fun inverse(x: Tensor): Tensor {
        require(isTrained) { "Scaler must be fitted before inverse transformation" }

        return TensorFactory.create(x.row, x.col) { i, j ->
            x[i, j] * std[0, j] + mean[0, j]
        }
    }
}
