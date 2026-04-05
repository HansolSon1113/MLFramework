package hs.ml.model.nn.sequential.recurrent.block

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.model.nn.ColConcatInputLayer
import hs.ml.model.nn.sequential.block.ContextAwareDecoderBlock
import hs.ml.model.nn.sequential.block.DecoderBlock
import hs.ml.model.nn.sequential.block.EncoderBlock
import hs.ml.model.nn.sequential.block.ManyToManyBlock

open class EncoderDecoderBlock(val encoderBlock: EncoderBlock, val decoderBlock: DecoderBlock, override val divider: Int) : ManyToManyBlock, ColConcatInputLayer {
    override fun forward(input: Node): Node {
        reset()
        val (source, targetInput) = split(input)

        val encoderOutput = encoderBlock.forward(source)
        transferState()

        val decoderInput: Node = if (decoderBlock is ContextAwareDecoderBlock) {
            decoderBlock.divider = encoderOutput.data.col
            encoderOutput.concat(targetInput, Tensor.Axis.VERTICAL)
        } else {
            targetInput
        }
        val decoderOutput = decoderBlock.forward(decoderInput)
        return decoderOutput
    }

    protected open fun reset() {
        encoderBlock.reset()
        decoderBlock.reset()
    }

    protected open fun split(input: Node): Pair<Node, Node> {
        return Pair(
            input.slice(0, divider, Tensor.Axis.VERTICAL),
            input.slice(divider, input.data.col, Tensor.Axis.VERTICAL)
        )
    }

    protected open fun transferState() {
        require(encoderBlock.states.size == decoderBlock.states.size) {
            "Encoder and Decoder must have the same state architecture"
        }
        decoderBlock.states = encoderBlock.states
    }


    override fun params(): List<Node> = encoderBlock.params() + decoderBlock.params()
}