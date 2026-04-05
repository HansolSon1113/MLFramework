package hs.ml.model.nn.sequential.recurrent.block

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.model.nn.Layer
import hs.ml.model.nn.sequential.block.ManyToOneBlock
import hs.ml.model.nn.sequential.recurrent.Recurrent

open class RecurrentManyToOneBlock(val rec: Recurrent): ManyToOneBlock {
    override fun forward(input: Node): Node {
        rec.reset()

        val length = input.data.col / rec.inputSize
        var output: Node? = null
        val steps = input.split(length, Tensor.Axis.VERTICAL)

        for (t in 0 until length) {
            output = rec.forward(steps[t])
        }

        return output ?: throw IllegalStateException("No output from recurrent layer")
    }

    override fun params(): List<Node> = rec.params()
}