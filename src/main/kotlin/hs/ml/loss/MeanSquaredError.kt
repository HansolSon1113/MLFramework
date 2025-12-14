package hs.ml.loss

import hs.ml.autograd.Node

class MeanSquaredError: Loss {
    override fun compute(yTrue: Node, yPred: Node): Node {
        require(yTrue.data.shape == yPred.data.shape) {
            "The number of samples in yTrue and yPred must be the same."
        }

        return (yTrue - yPred).pow(2).mean()
    }
}
