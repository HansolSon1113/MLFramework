package hs.ml.model.nn.sequential.recurrent.block.seq2seq

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.model.nn.Layer
import hs.ml.model.nn.sequential.block.DecoderBlock
import hs.ml.model.nn.sequential.recurrent.Recurrent

open class RecurrentDecoderBlock(val recurrent: Recurrent) : DecoderBlock {
    override var states: List<Node?>
        get() = recurrent.states
        set(value) { recurrent.states = value }

    override fun reset() {
        recurrent.reset()
    }

    override fun forward(input: Node): Node {
        val len = input.data.col / recurrent.inputSize
        val steps = input.split(len, Tensor.Axis.VERTICAL)
        var outputs: Node? = null
        var currentInput = steps[0]

        for (t in 0 until len) {
            val stepOutput = recurrent.forward(currentInput)

            if (outputs == null) {
                outputs = stepOutput
            } else {
                outputs = outputs.concat(stepOutput, Tensor.Axis.VERTICAL)
            }

            currentInput = stepOutput
        }
        return outputs ?: throw IllegalStateException("No output from decoder")
    }

    override fun params(): List<Node> = recurrent.params()
}