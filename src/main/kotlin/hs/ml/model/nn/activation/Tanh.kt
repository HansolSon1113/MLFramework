package hs.ml.model.nn.activation

import hs.ml.autograd.Node
import hs.ml.model.nn.Layer
import kotlin.math.exp

class Tanh : Layer() {
    override fun forward(input: Node): Node {
        return input.map(
            transform = { x ->
                val e = exp(2 * x)
                (e - 1) / (e + 1)
            },
            derivative = { x ->
                val e = exp(2 * x)
                val t = (e - 1) / (e + 1)
                1.0 - t * t
            }
        )
    }
}