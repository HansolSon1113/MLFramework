package hs.ml.model.nn.sequential.attention.scoring

import hs.ml.autograd.Node
import kotlin.math.sqrt

class ScaledDotProduct(colSize: Int) : DotProduct(colSize) {
    override fun score(st: Node, ht: Node): Node {
        val nSource = st.data.col.toDouble()

        return super.score(st, ht) * (1.0 / sqrt(nSource))
    }
}