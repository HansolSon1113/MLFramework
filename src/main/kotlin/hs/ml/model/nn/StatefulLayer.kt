package hs.ml.model.nn

import hs.ml.autograd.Node

interface StatefulLayer {
    val states: List<Node?>

    fun reset()
}