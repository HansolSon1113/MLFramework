package hs.ml.model.nn.sequential.recurrent

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.math.TensorFactory

class LSTM : Recurrent {
    val candidateActivation: (Node) -> Node

    constructor(inputSize: Int, hiddenSize: Int, activation: (Node) -> Node, candidateActivation: (Node) -> Node) :
            super(inputSize, hiddenSize, activation, 4) {
        this.candidateActivation = candidateActivation
    }

    override fun value(input: Node): Node {
        val state = states.getOrNull(0) ?: Node(TensorFactory.create(input.data.row, hiddenSize, 0.0))
        val cell = states.getOrNull(1) ?: Node(TensorFactory.create(input.data.row, hiddenSize, 0.0))
        val input = input * weights + state * hiddens + bias
        val (i, f, o, g) = input.split(divider, Tensor.Axis.VERTICAL)

        val nCell = activation(f).hadamard(cell) + activation(i).hadamard(candidateActivation(g))
        val nState = activation(o).hadamard(candidateActivation(nCell))
        states = listOf(nState, nCell)
        return nState
    }

    override fun reset() {
        super.reset()
    }
}