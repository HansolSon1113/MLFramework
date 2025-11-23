package hs.ml.model

import hs.ml.data.Tensor
import kotlin.math.sqrt

class StandardScaler : Scaler {
    private lateinit var mean: Tensor
    private lateinit var std: Tensor
    private var fitted = false

    override fun fit(x: Tensor, y: Tensor, epochs: Int, lr: Double) {
        // 평균 계산 (각 column 단위)
        mean = Tensor(1, x.col)
        std = Tensor(1, x.col)

        for (j in 0 until x.col) {
            var s = 0.0
            for (i in 0 until x.row) {
                s += x[i, j]
            }
            mean[0, j] = s / x.row
        }

        // 표준편차 계산
        for (j in 0 until x.col) {
            var acc = 0.0
            for (i in 0 until x.row) {
                val diff = x[i, j] - mean[0, j]
                acc += diff * diff
            }
            std[0, j] = sqrt(acc / x.row)

            // 혹시 std가 0이면 division-by-zero 방지
            if (std[0, j] == 0.0) std[0, j] = 1e-8
        }

        fitted = true
    }

    //수정 필요
    override var weights: Tensor
        get() = TODO("Not yet implemented")
        set(value) {}
    override var bias: Double
        get() = TODO("Not yet implemented")
        set(value) {}
    override var scaler: Scaler
        get() = TODO("Not yet implemented")
        set(value) {}
    override var epoch: Int
        get() = TODO("Not yet implemented")
        set(value) {}

    override fun predict(x: Tensor): Tensor {
        require(fitted) { "StandardScaler is not fitted yet" }
        require(x.col == mean.col) { "Column size mismatch" }

        val out = Tensor(x.row, x.col)

        for (i in 0 until x.row) {
            for (j in 0 until x.col) {
                out[i, j] = (x[i, j] - mean[0, j]) / std[0, j]
            }
        }

        return out
    }

    override fun inverseTransform(x: Tensor): Tensor {
        require(fitted) { "StandardScaler is not fitted yet" }

        val out = Tensor(x.row, x.col)
        for (i in 0 until x.row) {
            for (j in 0 until x.col) {
                out[i, j] = x[i, j] * std[0, j] + mean[0, j]
            }
        }
        return out
    }
}