package hs.ml.model.nn

import hs.ml.autograd.Node

interface Layer {
    fun forward(input: Node): Node
    fun params(): List<Node> = emptyList()
}