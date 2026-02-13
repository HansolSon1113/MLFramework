package hs.ml.model.regressor

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.math.TensorFactory
import hs.ml.model.nn.Dense

class LinearRegressor: Regressor() {
    private var layer: Dense? = null
    val weights: Tensor
        get() = layer?.weights?.data ?: TensorFactory.create(0, 0)

    val bias: Tensor
        get() = layer?.bias?.data ?: TensorFactory.create(0, 0)

    override fun forward(x: Node): Node {
        if (layer == null) {
            layer = Dense(x.data.col, 1)
        }

        return layer!!.forward(x)
    }

    override fun params(): List<Node> {
        return layer?.params() ?: emptyList()
    }

    override fun predict(x: Tensor): Tensor {
        val inputVal = Node(x)
        val outVal = forward(inputVal)
        return outVal.data
    }

    override fun toString(): String = "LinearRegressor(weights=$weights, bias=$bias, isTrained=$isTrained)"
}