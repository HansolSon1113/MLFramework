package hs.ml.model.nn.sequential.recurrent

import hs.ml.autograd.Node
import hs.ml.math.TensorFactory

class GRU(inputSize: Int, hiddenSize: Int, activation: (Node) -> Node) : Recurrent(inputSize, hiddenSize, activation, 3) {
    override fun value(input: Node): Node {
        val input = input * weights + state!! * hiddens + bias
        val (r, z, h) = input.split(3)
        val c = activation(h.hadamard(activation(r).hadamard(state!!)))
        return activation(z).hadamard(state!!) + (Node(TensorFactory.create(input.data.row, hiddenSize, 1.0)) - activation(z)).hadamard(c)
    }
}