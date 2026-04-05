package hs.ml.model.nn.block

import hs.ml.autograd.Node
import hs.ml.model.nn.Layer

class ResidualBlock(val layer: Layer): Layer {
    override fun forward(input: Node): Node = input + layer.forward(input)
}