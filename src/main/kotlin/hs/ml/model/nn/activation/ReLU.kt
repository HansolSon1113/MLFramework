package hs.ml.model.nn.activation

import hs.ml.autograd.Node
import hs.ml.model.nn.Layer
import kotlin.math.max

class ReLU : Layer {
    override fun forward(input: Node): Node {
        return input.map(
            transform = { x -> max(0.0, x) },
            derivative = { x -> if (x > 0.0) 1.0 else 0.0 }
        )
    }
}