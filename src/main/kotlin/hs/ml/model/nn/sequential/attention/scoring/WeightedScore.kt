package hs.ml.model.nn.sequential.attention.scoring

import hs.ml.autograd.Node
import hs.ml.math.TensorFactory

abstract class WeightedScore(val n: Int, val size: List<Pair<Int, Int>>, colSize: Int) : Score(colSize) {
    var w: List<Node> = List(n) {
        Node(
            TensorFactory.create(size[it].first, size[it].second)
        )
    }

    override fun params() = w
}