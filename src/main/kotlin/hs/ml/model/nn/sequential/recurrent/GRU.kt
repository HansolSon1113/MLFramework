package hs.ml.model.nn.sequential.recurrent

import hs.ml.autograd.Node
import hs.ml.math.TensorFactory

class GRU(inputSize: Int, hiddenSize: Int, activation: (Node) -> Node) :
    Recurrent(inputSize, hiddenSize * 4, activation) {

}