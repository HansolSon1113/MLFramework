package hs.ml.model.nn

import hs.ml.autograd.Node

class ResidualBlock(val layer: Layer): Layer() {
    override fun forward(input: Node): Node = input + layer.forward(input)
}