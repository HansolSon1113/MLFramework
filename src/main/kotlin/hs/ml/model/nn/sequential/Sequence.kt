package hs.ml.model.nn.sequential

import hs.ml.model.nn.Layer
import hs.ml.model.nn.StatefulLayer

interface Sequence : Layer {
    val inputSize: Int
    val outputSize: Int
}