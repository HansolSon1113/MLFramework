package hs.ml.model.nn.sequential.recurrent

import hs.ml.autograd.Node
import hs.ml.math.TensorFactory

class LSTM(inputSize: Int, hiddenSize: Int, activation: (Node) -> Node, val activationCandidate: (Node) -> Node) :
    Recurrent(inputSize, hiddenSize * 4, activation) {
    private var stateCell: Node? = null

    override fun value(input: Node): Node {
        if (stateCell == null) stateCell = Node(TensorFactory.create(input.data.row, hiddenSize, 0.0))

        val input = super.value(input)
        val (i, f, o, c) = input.split(4)

        stateCell = activation(f).hadamard(stateCell!!) + activation(i).hadamard(activationCandidate(c))
        return activation(o).hadamard(activationCandidate(stateCell!!))
    }

    override fun reset() {
        super.reset()
        stateCell = null
    }
}