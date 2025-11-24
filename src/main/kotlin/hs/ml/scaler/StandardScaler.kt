package hs.ml.scaler

import hs.ml.math.Tensor
import kotlin.math.sqrt

class StandardScaler : Scaler() {
    private lateinit var mean: Tensor
    private lateinit var std: Tensor

    override fun fit(x: Tensor, y: Tensor, epochs: Int, lr: Double) {
        val cols = x.col
        mean = Tensor(1, cols) { j, _ ->
            var sum = 0.0
            for (i in 0 until x.row) {
                sum += x[i, j]
            }
            sum / x.row
        }

        std = Tensor(1, cols) { j, _ ->
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
        require(isTrained) { "Scaler가 학습되지 않았습니다. fit 메서드를 먼저 호출하세요." }

        val result = Tensor(x.row, x.col) { i, j ->
            if (std[0, j] != 0.0) {
                (x[i, j] - mean[0, j]) / std[0, j]
            } else {
                0.0
            }
        }

        return result
    }

    override fun inverse(x: Tensor): Tensor {
        require(isTrained) { "Scaler가 학습되지 않았습니다. fit 메서드를 먼저 호출하세요." }

        val result = Tensor(x.row, x.col) { i, j ->
            x[i, j] * std[0, j] + mean[0, j]
        }

        return result
    }
}
