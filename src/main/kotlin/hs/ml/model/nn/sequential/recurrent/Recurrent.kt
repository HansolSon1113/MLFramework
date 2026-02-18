package hs.ml.model.nn.sequential.recurrent

import hs.ml.autograd.Node
import hs.ml.math.TensorFactory
import hs.ml.model.nn.Dense
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.sqrt

abstract class Recurrent : Dense {
    val hiddenSize: Int
        get() = outputSize
    val activation: (Node) -> Node
    var hiddens: Node
        private set
    protected var state: Node? = null

    constructor(inputSize: Int, hiddenSize: Int, activation: (Node) -> Node) : super(inputSize, hiddenSize) {
        require(inputSize > 0)
        require(hiddenSize > 0)

        this.activation = activation
        this.hiddens = Node(TensorFactory.create(hiddenSize, hiddenSize) { _, _ ->
            ThreadLocalRandom.current().nextGaussian() * sqrt(2.0 / hiddenSize)
        })
    }

    fun reset() {
        state = null
    }
}