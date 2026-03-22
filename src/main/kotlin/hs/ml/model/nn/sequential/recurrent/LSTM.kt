package hs.ml.model.nn.sequential.recurrent

import hs.ml.autograd.Node
import hs.ml.math.TensorFactory

class LSTM : Recurrent {
    val candidateActivation: (Node) -> Node
    private var stateCell: Node?

    constructor(inputSize: Int, hiddenSize: Int, activation: (Node) -> Node, candidateActivation: (Node) -> Node) :
            super(inputSize, hiddenSize, activation, 4) {
        this.candidateActivation = candidateActivation
        this.stateCell = null
    }

    override fun value(input: Node): Node {
        if (stateCell == null) stateCell = Node(TensorFactory.create(input.data.row, hiddenSize, 0.0))
        val input = input * weights + state!! * hiddens + bias
        val (i, f, o, g) = input.split(4)
        stateCell = activation(f).hadamard(stateCell!!) + activation(i).hadamard(candidateActivation(g))
        return activation(o).hadamard(candidateActivation(stateCell!!))
    }

    override fun reset() {
        super.reset()
        stateCell = null
    }
}