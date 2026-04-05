package hs.ml.model.nn.activation

import hs.ml.autograd.Node
import hs.ml.math.Tensor
import hs.ml.math.TensorFactory
import hs.ml.model.nn.Layer
import kotlin.math.exp

class Softmax(val axis: Tensor.Axis) : Layer {
    override fun forward(input: Node): Node {
        val max = input.data.max()
        val shf = input.data.map { it - max }
        val e = shf.map { exp(it) }
        val sum = e.sum(axis)

        val out = Node(
            TensorFactory.create(e.row, e.col) { i, j ->
                when (axis) {
                    Tensor.Axis.HORIZONTAL -> e[i, j] / sum[i, 0]
                    Tensor.Axis.VERTICAL -> e[i, j] / sum[0, j]
                }
            },
            children = listOf(input), operation = "softmax"
        )

        out._backward = {
            val gradS = out.grad.hadamard(out.data)
            val sumGradS = gradS.sum(axis)

            val localGrad = TensorFactory.create(out.data.row, out.data.col) { i, j ->
                val sumVal = when (axis) {
                    Tensor.Axis.HORIZONTAL -> sumGradS[i, 0]
                    Tensor.Axis.VERTICAL -> sumGradS[0, j]
                }
                out.data[i, j] * (out.grad[i, j] - sumVal)
            }

            input.grad = input.grad + localGrad
        }

        return out
    }
}