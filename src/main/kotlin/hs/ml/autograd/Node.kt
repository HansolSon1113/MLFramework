package hs.ml.autograd

import hs.ml.math.Tensor
import hs.ml.math.Tensor.Axis
import hs.ml.math.TensorFactory
import hs.ml.util.EPSILON
import kotlin.math.sqrt

data class Node(
    var data: Tensor,
    var grad: Tensor = TensorFactory.create(data.row, data.col, 0.0),
    val children: List<Node> = emptyList(),
    val operation: String = ""
) {
    internal var _backward: () -> Unit = {}

    operator fun plus(other: Node): Node {
        val out = Node(this.data + other.data, children = listOf(this, other), operation = "+")
        fun updateGrad(node: Node) {
            if (node.data.shape == out.grad.shape) {
                node.grad += out.grad
            } else if (node.data.row == 1 && node.data.col == out.grad.col) {
                node.grad += out.grad.sum(axis = Tensor.Axis.VERTICAL)
            }
        }

        out._backward = {
            updateGrad(this)
            updateGrad(other)
        }
        return out
    }

    operator fun minus(other: Node): Node = this + (-other)

    operator fun unaryMinus(): Node = this * -1.0

    operator fun times(other: Node): Node {
        val out = Node(this.data * other.data, children = listOf(this, other), operation = "*")

        out._backward = {
            this.grad = this.grad + (out.grad * other.data.T)
            other.grad = other.grad + (this.data.T * out.grad)
        }
        return out
    }

    operator fun times(scalar: Double): Node {
        val out = Node(this.data * scalar, children = listOf(this), operation = "*$scalar")

        out._backward = {
            this.grad = this.grad + (out.grad * scalar)
        }
        return out
    }

    infix fun hadamard(other: Node): Node {
        val out = Node(this.data hadamard other.data, children = listOf(this, other), operation = "hadamard")

        out._backward = {
            this.grad = this.grad + (out.grad hadamard other.data)
            other.grad = other.grad + (out.grad hadamard this.data)
        }
        return out
    }

    fun map(transform: (Double) -> Double, derivative: (Double) -> Double): Node {
        val outData = this.data.map(transform)
        val out = Node(outData, children = listOf(this), operation = "map")

        out._backward = {
            val localGrad = this.data.map(derivative)
            this.grad = this.grad + (out.grad.hadamard(localGrad))
        }
        return out
    }

    fun pow(exponent: Int): Node {
        val out = Node(this.data.pow(exponent), children = listOf(this), operation = "^$exponent")

        out._backward = {
            val n = exponent.toDouble()
            val localDerivative = this.data.pow(exponent - 1) * n
            this.grad = this.grad + (out.grad.hadamard(localDerivative))
        }
        return out
    }

    fun sum(axis: Axis): Node {
        val out = Node(this.data.sum(axis), children = listOf(this), operation = "sum_$axis")
        out._backward = {
            when (axis) {
                Tensor.Axis.VERTICAL -> {
                    for (i in 0 until this.grad.row) {
                        for (j in 0 until this.grad.col) {
                            this.grad[i, j] += out.grad[0, j]
                        }
                    }
                }
                Tensor.Axis.HORIZONTAL -> {
                    for (i in 0 until this.grad.row) {
                        for (j in 0 until this.grad.col) {
                            this.grad[i, j] += out.grad[i, 0]
                        }
                    }
                }
            }
        }
        return out
    }

    fun mean(): Node {
        val totalElements = (this.data.row * this.data.col).toDouble()
        var sum = 0.0
        for (i in 0 until this.data.row) {
            for (j in 0 until this.data.col) {
                sum += this.data[i, j]
            }
        }
        val out = Node(TensorFactory.create(1, 1, sum / totalElements), children = listOf(this), operation = "mean")

        out._backward = {
            val gradVal = out.grad[0, 0] / totalElements
            this.grad = this.grad + TensorFactory.create(this.data.row, this.data.col, gradVal)
        }
        return out
    }

    fun log(): Node = this.map({ kotlin.math.ln(it) }, { 1.0 / it })

    fun split(parts: Int, axis: Tensor.Axis): Array<Node> {
        require(parts > 0) { "Invalid parts: $parts" }

        return when (axis) {
            Tensor.Axis.VERTICAL -> {
                val c = data.col / parts
                Array(parts) { i ->
                    val startCol = i * c
                    Node(
                        data.slice(startCol, startCol + c, Tensor.Axis.VERTICAL),
                        children = listOf(this),
                        operation = "split_VERT[$i/$parts]"
                    ).also { node ->
                        node._backward = {
                            for (row in 0 until grad.row) {
                                for (col in 0 until node.grad.col) {
                                    grad[row, startCol + col] += node.grad[row, col]
                                }
                            }
                        }
                    }
                }
            }

            Tensor.Axis.HORIZONTAL -> {
                val r = data.row / parts
                Array(parts) { i ->
                    val startRow = i * r
                    Node(
                        data.slice(startRow, startRow + r, Tensor.Axis.HORIZONTAL),
                        children = listOf(this),
                        operation = "split_HORZ[$i/$parts]"
                    ).also { node ->
                        node._backward = {
                            for (row in 0 until node.grad.row) {
                                for (col in 0 until grad.col) {
                                    grad[startRow + row, col] += node.grad[row, col]
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fun slice(start: Int, end: Int, axis: Tensor.Axis): Node {
        val outData = this.data.slice(start, end, axis)
        val out = Node(outData, children = listOf(this), operation = "slice")

        out._backward = {
            when (axis) {
                Tensor.Axis.VERTICAL -> {
                    for (row in 0 until out.grad.row) {
                        for (col in 0 until out.grad.col) {
                            this.grad[row, start + col] += out.grad[row, col]
                        }
                    }
                }
                Tensor.Axis.HORIZONTAL -> {
                    for (row in 0 until out.grad.row) {
                        for (col in 0 until out.grad.col) {
                            this.grad[start + row, col] += out.grad[row, col]
                        }
                    }
                }
            }
        }
        return out
    }

    fun concat(other: Node, axis: Tensor.Axis): Node {
        val outData = this.data.concat(other.data, axis)
        val out = Node(outData, children = listOf(this, other), operation = "concat_$axis")

        out._backward = {
            when (axis) {
                Tensor.Axis.VERTICAL -> {
                    val gradThis = out.grad.slice(0, this.data.col, Tensor.Axis.VERTICAL)
                    val gradOther = out.grad.slice(this.data.col, out.data.col, Tensor.Axis.VERTICAL)

                    this.grad = this.grad + gradThis
                    other.grad = other.grad + gradOther
                }

                Tensor.Axis.HORIZONTAL -> {
                    val gradThis = out.grad.slice(0, this.data.row, Tensor.Axis.HORIZONTAL)
                    val gradOther = out.grad.slice(this.data.row, out.data.row, Tensor.Axis.HORIZONTAL)

                    this.grad = this.grad + gradThis
                    other.grad = other.grad + gradOther
                }
            }
        }
        return out
    }

    fun transpose(): Node {
        val out = Node(this.data.T, children = listOf(this), operation = "transpose")
        out._backward = {
            this.grad = this.grad + out.grad.T
        }
        return out
    }

    fun cosineSimilarity(other: Node): Node {
        val numerator = this * other.transpose()

        fun norm(node: Node) = node.pow(2)
            .sum(Tensor.Axis.HORIZONTAL)
            .map({sqrt(it)}, {0.5 / sqrt(it)})

        val denominator = norm(this) * norm(other).transpose()
        val eps = denominator.map({ it + EPSILON }, { 1.0 })
        val inverseDenominator = eps.map({ 1.0 / it }, { -1.0 / (it * it) })

        return numerator.hadamard(inverseDenominator)
    }

    fun backward(initialGrad: Tensor? = null) {
        val topo = mutableListOf<Node>()
        val visited = mutableSetOf<Node>()

        fun visit(v: Node) {
            if (v !in visited) {
                visited.add(v)
                for (child in v.children) visit(child)
                topo.add(v)
            }
        }
        visit(this)

        this.grad = initialGrad ?: TensorFactory.create(data.row, data.col, 1.0)

        for (node in topo.reversed()) {
            node._backward()
        }
    }
}