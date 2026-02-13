package hs.ml.train.optimizer

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.math.TensorFactory
import kotlin.math.pow
import kotlin.math.sqrt

class Adam : Optimizer {
    override var lr: Double
        private set
    var beta: DoubleArray
        private set
    var epsilon: Double
        private set

    private data class AdamState(
        val m: Tensor,
        val v: Tensor
    )

    private val state = mutableMapOf<Node, AdamState>()

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

    override fun step(params: List<Node>) {
        t++

        val b1 = beta[0]
        val b2 = beta[1]

        val correction1 = 1.0 - b1.pow(t)
        val correction2 = 1.0 - b2.pow(t)

        for (param in params) {
            if (!state.containsKey(param)) {
                state[param] = AdamState(
                    m = TensorFactory.create(param.data.row, param.data.col, 0.0),
                    v = TensorFactory.create(param.data.row, param.data.col, 0.0)
                )
            }

            val s = state[param]!!
            val m = s.m
            val v = s.v
            val grad = param.grad
            val data = param.data

            for (i in 0 until data.row) {
                for (j in 0 until data.col) {
                    val g = grad[i, j]

                    m[i, j] = b1 * m[i, j] + (1 - b1) * g
                    v[i, j] = b2 * v[i, j] + (1 - b2) * (g * g)

                    val mHat = m[i, j] / correction1
                    val vHat = v[i, j] / correction2

                    data[i, j] = data[i, j] - lr * (mHat / (sqrt(vHat) + epsilon))
                }
            }
        }
    }
}