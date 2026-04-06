package hs.ml.model.nn

import hs.ml.autograd.Node

class Identity : Layer {
    override fun forward(input: Node): Node = input

    override fun params(): List<Node> = emptyList()
}