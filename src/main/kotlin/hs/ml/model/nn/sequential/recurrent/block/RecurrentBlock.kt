package hs.ml.model.nn.sequential.recurrent.block

import hs.ml.autograd.Node
import hs.ml.model.nn.Layer
import hs.ml.model.nn.sequential.recurrent.Recurrent

class RecurrentBlock(val rec: Recurrent): Layer() {
    override fun forward(input: Node): Node {
        rec.reset()

        val length = input.data.col / rec.inputSize
        var output: Node? = null

        for (t in 0 until length) {
            val step = Node(input.data.slice(t * rec.inputSize, (t + 1) * rec.inputSize))
            output = rec.forward(step)
        }

        return output ?: throw IllegalStateException("No output from recurrent layer")
    }

    override fun params(): List<Node> = rec.params()
}