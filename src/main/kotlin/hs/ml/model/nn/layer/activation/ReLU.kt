package hs.ml.model.nn.layer.activation

import hs.ml.math.Tensor
import hs.ml.model.nn.layer.Layer
import kotlin.math.max

class ReLU : Layer() {
    override fun forward(input: Tensor): Tensor {
        this.cache = input
        return input.map { x -> max(0.0, x) }
    }
}