package hs.ml.model.nn.sequential.attention.scoring

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.model.nn.ColConcatInputLayer

abstract class Score(override val divider: Int) : ColConcatInputLayer {
    override fun forward(input: Node): Node {
        return score(
            input.slice(0, colSize, Tensor.Axis.VERTICAL),
            input.slice(colSize, input.data.col, Tensor.Axis.VERTICAL)
        )
    }

    abstract fun score(st: Node, ht: Node): Node
}