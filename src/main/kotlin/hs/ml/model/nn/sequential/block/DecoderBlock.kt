package hs.ml.model.nn.sequential.block

import hs.ml.autograd.Node
import hs.ml.model.nn.Layer
import hs.ml.model.nn.StatefulLayer

interface DecoderBlock : StatefulLayer, Layer {
    override var states: List<Node?>
}