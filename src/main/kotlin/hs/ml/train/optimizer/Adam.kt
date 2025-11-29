package hs.ml.train.optimizer

import hs.ml.math.Tensor
import kotlin.math.pow
import kotlin.math.sqrt

class Adam : Optimizer {
    override var lr: Double
        private set
    var beta: DoubleArray
        private set
    var epsilon: Double
        private set
    private var m: DoubleArray? = null
    private var v: DoubleArray? = null
    private var t: Int = 0

    constructor(lr: Double = 0.01, b1: Double = 0.9, b2: Double = 0.99, epsilon: Double = 1e-8) {
        this.lr = lr
        this.beta = doubleArrayOf(b1, b2)
        this.epsilon = epsilon
    }

    override fun step(params: Tensor, gradients: Tensor): Tensor {
        val size = params.row * params.col

        if (m == null || m!!.size != size) m = DoubleArray(size)
        if (v == null || v!!.size != size) v = DoubleArray(size)

        t++

        val b1 = beta[0]
        val b2 = beta[1]

        val newFlatData = DoubleArray(size)

        for (r in 0 until params.row) {
            for (c in 0 until params.col) {
                val idx = r * params.col + c

                val theta = params[r, c]
                val g = gradients[r, c]

                m!![idx] = b1 * m!![idx] + (1.0 - b1) * g
                v!![idx] = b2 * v!![idx] + (1.0 - b2) * (g * g)
                val mHat = m!![idx] / (1.0 - b1.pow(t))
                val vHat = v!![idx] / (1.0 - b2.pow(t))

                newFlatData[idx] = theta - (lr * mHat) / (sqrt(vHat) + epsilon)
            }
        }

        return Tensor(params.row, params.col) { r, c ->
            newFlatData[r * params.col + c]
        }
    }
}