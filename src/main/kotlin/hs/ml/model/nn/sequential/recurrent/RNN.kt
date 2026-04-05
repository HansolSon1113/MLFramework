package hs.ml.model.nn.sequential.recurrent

import hs.ml.autograd.Node

class RNN(inputSize: Int, hiddenSize: Int, activation: (Node) -> Node) : Recurrent(inputSize, hiddenSize, activation, 1) {
    override fun value(input: Node): Node = activation(input * weights + states.first()!! * hiddens + bias)
}

