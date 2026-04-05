package hs.ml.model.nn.sequential.recurrent.block

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.model.nn.Layer
import hs.ml.model.nn.sequential.block.ManyToOneBlock
import hs.ml.model.nn.sequential.recurrent.Recurrent

open class RecurrentBidirectionalManyToOneBlock(val recForward: Recurrent, val recBackward: Recurrent): ManyToOneBlock {
    override fun forward(input: Node): Node {
        recForward.reset()
        recBackward.reset()

        val length = input.data.col / recForward.inputSize
        var outForward: Node? = null
        var outBackward: Node? = null
        val steps = input.split(length, Tensor.Axis.VERTICAL)

        for (t in 0 until length) {
            outForward = recForward.forward(steps[t])
        }

        for (t in length -1 downTo 0) {
            outBackward = recBackward.forward(steps[t])
        }

        val finalForward = outForward ?: throw IllegalStateException("No output from forward recurrent layer")
        val finalBackward = outBackward ?: throw IllegalStateException("No output from backward recurrent layer")

        return finalForward.concat(finalBackward, Tensor.Axis.VERTICAL)
    }

    override fun params(): List<Node> = recForward.params() + recBackward.params()
}