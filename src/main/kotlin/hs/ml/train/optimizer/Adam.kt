package hs.ml.train.optimizer

import hs.ml.math.Tensor
import kotlin.math.sqrt

class Adam : Optimizer {
    override var lr: Double
        private set
    var beta: DoubleArray
        private set
    var epsilon: Double
        private set

    // First moment
    private var mW: Tensor? = null
    private var mB: Double? = null

    // Second moment
    private var vW: Tensor? = null
    private var vB: Double? = null

    private var t: Int = 0

    constructor(
        lr: Double = 0.01,
        b1: Double = 0.9,
        b2: Double = 0.99,
        epsilon: Double = 1e-8
    ) {
        this.lr = lr
        this.beta = doubleArrayOf(b1, b2)
        this.epsilon = epsilon
    }

    override fun step(
        params: Pair<Tensor, Double>,
        gradients: Pair<Tensor, Double>
    ): Pair<Tensor, Double> {

        val (w, b) = params
        val (gw, gb) = gradients
        val b1 = beta[0]
        val b2 = beta[1]

        t++

        // ===== Lazy init for m, v =====
        if (mW == null) {
            mW = Tensor(w.shape.first, w.shape.second)
            vW = Tensor(w.shape.first, w.shape.second)
            mB = 0.0
            vB = 0.0
        }

        val mW = this.mW!!
        val vW = this.vW!!
        var mB = this.mB!!
        var vB = this.vB!!

        // ===== Update m and v for weights =====
        for (i in 0 until w.shape.first) {
            for (j in 0 until w.shape.second) {
                val g = gw[i, j]

                mW[i, j] = b1 * mW[i, j] + (1 - b1) * g
                vW[i, j] = b2 * vW[i, j] + (1 - b2) * (g * g)
            }
        }

        // bias
        mB = b1 * mB + (1 - b1) * gb
        vB = b2 * vB + (1 - b2) * (gb * gb)

        // ===== Bias correction =====
        val mW_hat = Tensor(w.shape.first, w.shape.second)
        val vW_hat = Tensor(w.shape.first, w.shape.second)

        for (i in 0 until w.shape.first) {
            for (j in 0 until w.shape.second) {
                mW_hat[i, j] = mW[i, j] / (1 - Math.pow(b1, t.toDouble()))
                vW_hat[i, j] = vW[i, j] / (1 - Math.pow(b2, t.toDouble()))
            }
        }

        val mB_hat = mB / (1 - Math.pow(b1, t.toDouble()))
        val vB_hat = vB / (1 - Math.pow(b2, t.toDouble()))

        // ===== Parameter update =====
        val newW = Tensor(w.shape.first, w.shape.second)
        for (i in 0 until w.shape.first) {
            for (j in 0 until w.shape.second) {
                newW[i, j] =
                    w[i, j] - lr * (mW_hat[i, j] / (sqrt(vW_hat[i, j]) + epsilon))
            }
        }

        val newB = b - lr * (mB_hat / (sqrt(vB_hat) + epsilon))

        // ===== Save updated states =====
        this.mW = mW
        this.vW = vW
        this.mB = mB
        this.vB = vB

        return Pair(newW, newB)
    }
}