package hs.ml.model.nn.sequential.recurrent

import hs.ml.autograd.Node
import hs.ml.math.TensorFactory
import hs.ml.model.nn.Layer
import hs.ml.model.nn.StatefulLayer
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.sqrt

abstract class Recurrent : Layer, StatefulLayer {
    val inputSize: Int
    val hiddenSize: Int
    val outputSize: Int
        get() = hiddenSize
    val activation: (Node) -> Node
    var weights: Node
        private set
    var hiddens: Node
        private set
    var bias: Node
        private set
    protected var state: Node?

    constructor(inputSize: Int, hiddenSize: Int, activation: (Node) -> Node, gates: Int) : super() {
        require(inputSize > 0 && hiddenSize > 0)

        this.inputSize = inputSize
        this.hiddenSize = hiddenSize
        this.activation = activation
        this.weights = Node(TensorFactory.create(inputSize, hiddenSize * gates) { _, _ ->
            ThreadLocalRandom.current().nextGaussian() * sqrt(2.0 / inputSize)
        })
        this.hiddens = Node(TensorFactory.create(hiddenSize, hiddenSize * gates) { _, _ ->
            ThreadLocalRandom.current().nextGaussian() * sqrt(2.0 / hiddenSize)
        })
        this.bias = Node(TensorFactory.create(1, hiddenSize * gates, 0.0))
        this.state = null
    }

    override fun forward(input: Node): Node {
        require(state == null || state!!.data.row == input.data.row) { "Batch size between state and input does not match" }
        if (state == null) state = Node(TensorFactory.create(input.data.row, hiddenSize, 0.0))
        state = value(input)
        return state!!
    }

    override fun params(): List<Node> = listOf(weights, hiddens, bias)

    protected abstract fun value(input: Node): Node

    override fun reset() {
        state = null
    }
}