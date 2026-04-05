package hs.ml.model.nn.sequential.recurrent

import hs.ml.autograd.Node
import hs.ml.math.TensorFactory
import hs.ml.model.nn.ColConcatInputLayer
import hs.ml.model.nn.StatefulLayer
import hs.ml.model.nn.sequential.Sequence
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.sqrt

abstract class Recurrent(
    override val inputSize: Int,
    val hiddenSize: Int,
    val activation: (Node) -> Node,
    val gates: Int,
) : Sequence, StatefulLayer, ColConcatInputLayer {

    init {
        require(inputSize > 0 && hiddenSize > 0)
    }

    override val outputSize: Int
        get() = hiddenSize

    override val divider: Int
        get() = gates

    override var states: List<Node?> = emptyList()

    var weights: Node = Node(TensorFactory.create(inputSize, hiddenSize * gates) { _, _ ->
        ThreadLocalRandom.current().nextGaussian() * sqrt(2.0 / inputSize)
    })
        private set

    var hiddens: Node = Node(TensorFactory.create(hiddenSize, hiddenSize * gates) { _, _ ->
        ThreadLocalRandom.current().nextGaussian() * sqrt(2.0 / hiddenSize)
    })
        private set

    var bias: Node = Node(TensorFactory.create(1, hiddenSize * gates, 0.0))
        private set

    override fun forward(input: Node): Node {
        val hiddenState = states.firstOrNull()

        require(hiddenState == null || hiddenState.data.row == input.data.row) {
            "Batch size between state and input does not match"
        }

        return value(input)
    }

    override fun params(): List<Node> = listOf(weights, hiddens, bias)

    protected abstract fun value(input: Node): Node

    override fun reset() {
        states = emptyList()
    }
}