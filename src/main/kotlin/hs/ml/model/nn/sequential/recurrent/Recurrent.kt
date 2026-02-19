package hs.ml.model.nn.sequential.recurrent

import hs.ml.autograd.Node
import hs.ml.math.TensorFactory
import hs.ml.model.nn.Dense
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.sqrt

open class Recurrent : Dense {
    val hiddenSize: Int
        get() = outputSize
    val activation: (Node) -> Node
    var hiddens: Node
        private set
    protected var state: Node?

    constructor(inputSize: Int, hiddenSize: Int, activation: (Node) -> Node) : super(inputSize, hiddenSize) {
        require(inputSize > 0)
        require(hiddenSize > 0)

        this.activation = activation
        this.hiddens = Node(TensorFactory.create(hiddenSize, hiddenSize) { _, _ ->
            ThreadLocalRandom.current().nextGaussian() * sqrt(2.0 / hiddenSize)
        })

        this.state = null
    }

    override fun forward(input: Node): Node {
        require(state == null || state!!.data.row == input.data.row) { "Batch size between state and input does not match" }

        if (state == null) state = Node(TensorFactory.create(input.data.row, hiddenSize, 0.0))

        state = activation(value(input))
        return state!!
    }

    override fun params(): List<Node> = listOf(weights, hiddens, bias)

    open fun value(input: Node): Node = input * weights + state!! * hiddens + bias

    open fun reset() {
        state = null
    }
}