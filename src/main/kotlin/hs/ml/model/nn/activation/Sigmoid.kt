package hs.ml.model.nn.activation

import hs.ml.autograd.Node
import hs.ml.model.nn.Layer
import kotlin.math.exp

class Sigmoid : Layer {
    override fun forward(input: Node): Node {
        return input.map(
            transform = { x ->
                1.0 / (1.0 + exp(-x))
            },
            derivative = { x ->
                val s = 1.0 / (1.0 + exp(-x))
                s * (1.0 - s)
            }
        )
    }
}