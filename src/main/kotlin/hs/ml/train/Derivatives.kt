package hs.ml.train

import kotlin.math.exp
import kotlin.math.tanh as th
import kotlin.math.pow

class Derivatives {
    companion object {
        val mse: (Double) -> Double = { diff ->
            2 * diff
        }

        val mae: (Double) -> Double = { diff ->
            when {
                diff > 0 -> 1.0
                diff < 0 -> -1.0
                else -> 0.0
            }
        }

        val sigmoid: (Double) -> Double = { x ->
            val s = 1.0 / (1.0 + exp(-x))
            s * (1.0 - s)
        }

        //For activated value
        val activatedSigmoid: (Double) -> Double = { y ->
            y * (1.0 - y)
        }

        val tanh: (Double) -> Double = { x ->
            1.0 - th(x).pow(2)
        }

        val relu: (Double) -> Double = { x ->
            if (x > 0.0) 1.0 else 0.0
        }

        val leakyRelu: (Double) -> Double = { x ->
            if (x > 0.0) 1.0 else 0.01
        }

        val linear: (Double) -> Double = { _ ->
            1.0
        }
    }
}