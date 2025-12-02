package hs.ml.model.nn.layer.activation

import hs.ml.math.Tensor
import hs.ml.model.nn.layer.Layer
import kotlin.math.exp

class Sigmoid : Layer() {
    override fun forward(input: Tensor): Tensor {
        this.cache = input
        return input.map { x -> 1.0 / (1.0 + exp(-x)) }
    }
}