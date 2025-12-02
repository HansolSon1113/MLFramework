package hs.ml.model.nn.layer

import hs.ml.math.Tensor

abstract class Layer {
    protected var cache: Tensor? = null

    abstract fun forward(input: Tensor): Tensor
    open fun getParams(): List<Tensor> = emptyList()
}