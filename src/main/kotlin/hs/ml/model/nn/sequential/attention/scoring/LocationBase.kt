package hs.ml.model.nn.sequential.attention.scoring

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.model.nn.activation.Softmax

//hSize should be output col size!
class LocationBase(sSize: Int, hSize: Int, colSize: Int) : WeightedScore(1,
    listOf(Pair(sSize, hSize * 2))
    , colSize
) {
    private val softmax = Softmax(Tensor.Axis.HORIZONTAL)

    override fun score(st: Node, ht: Node): Node = softmax.forward(w[0] * st)
}