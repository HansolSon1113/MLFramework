package hs.ml.loss

import hs.ml.autograd.Node

interface Loss {
    fun compute(yTrue: Node, yPred: Node): Node
}
