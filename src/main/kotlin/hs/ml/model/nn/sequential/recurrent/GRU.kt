package hs.ml.model.nn.sequential.recurrent

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.math.TensorFactory

class GRU(inputSize: Int, hiddenSize: Int, activation: (Node) -> Node) : Recurrent(inputSize, hiddenSize, activation, 3) {
    override fun value(input: Node): Node {
        val input = input * weights + states.first()!! * hiddens + bias
        val (r, z, h) = input.split(divider, Tensor.Axis.VERTICAL)
        val c = activation(h.hadamard(activation(r).hadamard(states.first()!!)))

        val nState = activation(z).hadamard(states.first()!!) + (Node(TensorFactory.create(input.data.row, hiddenSize, 1.0)) - activation(z)).hadamard(c)
        states = listOf(nState)
        return nState
    }
}