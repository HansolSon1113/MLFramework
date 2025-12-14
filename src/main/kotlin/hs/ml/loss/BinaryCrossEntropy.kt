package hs.ml.loss

import hs.ml.autograd.Node
import hs.ml.math.Tensor

class BinaryCrossEntropy: Loss {
    override fun compute(yTrue: Node, yPred: Node): Node {
        require(yTrue.data.shape == yPred.data.shape) {
            "The number of samples in yTrue and yPred must be the same."
        }

        val rows = yTrue.data.row
        val cols = yTrue.data.col
        val epsVal = 1e-15

        val ones = Node(Tensor(rows, cols, 1.0))
        val epsilon = Node(Tensor(rows, cols, epsVal))

        val logP = (yPred + epsilon).log()
        val logOneMinusP = (ones - yPred + epsilon).log()

        val term1 = yTrue * logP
        val term2 = (ones - yTrue) * logOneMinusP

        return -(term1 + term2).mean()
    }
}