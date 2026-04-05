package hs.ml.model.nn.sequential.attention.scoring

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.model.nn.activation.Tanh

class Additive(sSize: Int, hSize: Int, attSize: Int, colSize: Int) : WeightedScore(2,
    listOf(
        Pair(sSize, attSize),
        Pair(hSize, attSize),
        Pair(attSize, 1)
    ), colSize
) {
    private val tanh = Tanh()

    override fun score(st: Node, ht: Node): Node {
        val projQ = st * w[0]
        val projK = ht * w[1]
        val n = st.data.row
        val queryRows = projQ.split(n, Tensor.Axis.HORIZONTAL)

        var mat: Node? = null

        for (i in 0 until n) {
            val sumNode = projK + queryRows[i]

            val activated = tanh.forward(sumNode)
            val scoreRow = activated * w[2]

            val scoreTransposed = scoreRow.transpose()

            if (mat == null) {
                mat = scoreTransposed
            } else {
                mat = mat.concat(scoreTransposed, Tensor.Axis.HORIZONTAL)
            }
        }

        return mat!!
    }
}