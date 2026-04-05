package hs.ml.model.nn.sequential.attention.scoring

import hs.ml.autograd.Node

class General(sSize: Int, hSize: Int, colSize: Int) : WeightedScore(1,
    listOf(Pair(sSize, hSize))
    , colSize
) {
    override fun score(st: Node, ht: Node): Node {
        return st * w[0] * ht.transpose()
    }
}