package hs.ml.model.nn.sequential.attention.block

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.math.TensorFactory
import hs.ml.model.nn.ColConcatInputLayer
import hs.ml.model.nn.Layer
import hs.ml.model.nn.activation.Softmax
import hs.ml.model.nn.sequential.attention.scoring.Score
import hs.ml.model.nn.sequential.block.ContextAwareDecoderBlock
import hs.ml.model.nn.sequential.block.DecoderBlock
import hs.ml.model.nn.sequential.recurrent.Recurrent

open class RecurrentAttentionDecoderBlock(val recurrent: Recurrent, val scorer: Score,
                                          val encoderHiddenSize: Int, val targetSize: Int,
                                          override var divider: Int = 0
) : ContextAwareDecoderBlock, ColConcatInputLayer {
    override var states: List<Node?>
        get() = recurrent.states
        set(value) { recurrent.states = value }

    private val softmax = Softmax(Tensor.Axis.HORIZONTAL)

    override fun reset() {
        recurrent.reset()
    }

    override fun forward(input: Node): Node {
        require(divider > 0) { "encoderOutputCols must be set by EncoderDecoderBlock before calling forward" }

        val encoderOutputsConcat = input.slice(0, divider, Tensor.Axis.VERTICAL)
        val targetInput = input.slice(divider, input.data.col, Tensor.Axis.VERTICAL)
        val srcLength = encoderOutputsConcat.data.col / encoderHiddenSize
        val encoderOutputs = encoderOutputsConcat.split(srcLength, Tensor.Axis.VERTICAL)
        val targetLength = targetInput.data.col / targetSize
        val targetSteps = targetInput.split(targetLength, Tensor.Axis.VERTICAL)
        var outputs: Node? = null
        var currentStepInput = targetSteps[0]

        for (t in 0 until targetLength) {
            val stPrev = states.firstOrNull() ?: Node(TensorFactory.create(input.data.row, recurrent.hiddenSize, 0.0))
            var mat: Node? = null

            for (tp in 0 until srcLength) {
                val htp = encoderOutputs[tp]
                val score = scorer.score(stPrev, htp)

                mat = if (mat == null) score else mat.concat(score, Tensor.Axis.VERTICAL)
            }

            val alphas = softmax.forward(mat!!)
            val alphaCols = alphas.split(srcLength, Tensor.Axis.VERTICAL)
            var ct: Node? = null

            for (tp in 0 until srcLength) {
                val atp = alphaCols[tp]
                val htp = encoderOutputs[tp]

                var broadcastedAlpha = atp
                for (c in 1 until htp.data.col) {
                    broadcastedAlpha = broadcastedAlpha.concat(atp, Tensor.Axis.VERTICAL)
                }

                val weightedH = htp.hadamard(broadcastedAlpha)
                ct = if (ct == null) weightedH else ct + weightedH
            }

            val decoderInput = currentStepInput.concat(ct!!, Tensor.Axis.VERTICAL)
            val stepOutput = recurrent.forward(decoderInput)

            if (outputs == null) {
                outputs = stepOutput
            } else {
                outputs = outputs.concat(stepOutput, Tensor.Axis.VERTICAL)
            }
            
            currentStepInput = stepOutput
        }
        return outputs ?: throw IllegalStateException("No output from decoder")
    }

    override fun params(): List<Node> = recurrent.params() + scorer.params()
}