package hs.ml.model.nn.sequential.block

import hs.ml.model.nn.ColConcatInputLayer

interface ContextAwareDecoderBlock : DecoderBlock, ColConcatInputLayer {
    override var divider: Int
}