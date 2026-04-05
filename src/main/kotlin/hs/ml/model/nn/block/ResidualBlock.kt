package hs.ml.model.nn.block

import hs.ml.autograd.Node
import hs.ml.model.nn.Layer

class ResidualBlock(val fx: Layer, val hx: Layer? = null) : Layer {
    override fun forward(input: Node): Node {
        val shortcut = hx?.forward(input) ?: input

        return shortcut + fx.forward(input)
    }

    override fun params(): List<Node> = fx.params() + (hx?.params() ?: emptyList())
}