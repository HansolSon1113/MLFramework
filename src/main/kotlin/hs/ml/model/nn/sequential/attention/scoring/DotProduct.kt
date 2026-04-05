package hs.ml.model.nn.sequential.attention.scoring

import hs.ml.autograd.Node

open class DotProduct(colSize: Int) : Score(colSize) {
    override fun score(st: Node, ht: Node): Node = st.transpose() * ht
}