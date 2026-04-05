package hs.ml.model.nn.sequential.recurrent.block.seq2seq

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.model.nn.Layer
import hs.ml.model.nn.sequential.block.EncoderBlock
import hs.ml.model.nn.sequential.recurrent.Recurrent

open class RecurrentEncoderBlock(val recurrent: Recurrent) : EncoderBlock {
    override var states: List<Node?>
        get() = recurrent.states
        set(value) { recurrent.states = value }

    override fun reset() {
        recurrent.reset()
    }

    override fun forward(input: Node): Node {
        val srcLength = input.data.col / recurrent.inputSize
        val srcSteps = input.split(srcLength, Tensor.Axis.VERTICAL)
        var outputs: Node? = null

        for (t in 0 until srcLength) {
            val stepOutput = recurrent.forward(srcSteps[t])
            if (outputs == null) {
                outputs = stepOutput
            } else {
                outputs = outputs.concat(stepOutput, Tensor.Axis.VERTICAL)
            }
        }

        return outputs ?: throw IllegalStateException("No output from encoder")
    }

    override fun params(): List<Node> = recurrent.params()
}