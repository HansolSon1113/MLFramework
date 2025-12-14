package hs.ml.model

import hs.ml.autograd.Node

abstract class Model {
    var param: ModelParameter = ModelParameter()
    var epoch: Int = 0
    val isTrained: Boolean
        get() = epoch > 0

    abstract fun params(): List<Node>
    abstract fun forward(x: Node): Node
}
