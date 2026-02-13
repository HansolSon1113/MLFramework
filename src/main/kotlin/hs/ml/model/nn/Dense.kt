package hs.ml.model.nn

import java.util.concurrent.ThreadLocalRandom
import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.math.TensorFactory
import kotlin.math.sqrt

class Dense : Layer {
    val inputSize: Int
    val outputSize: Int

    constructor(inputSize: Int, outputSize: Int) : super() {
        require(inputSize > 0)
        require(outputSize > 0)

        this.inputSize = inputSize
        this.outputSize = outputSize
        this.weights = Node(TensorFactory.create(inputSize, outputSize) { _, _ ->
            ThreadLocalRandom.current().nextGaussian() * sqrt(2.0 / inputSize)
        })
        this.bias = Node(TensorFactory.create(1, outputSize, 0.0))
    }

    var weights: Node
        private set
    var bias: Node
        private set

    override fun forward(input: Node): Node {
        return (input * weights) + bias
    }

    override fun params(): List<Node> = listOf(weights, bias)
}