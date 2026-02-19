package hs.ml.autograd

import hs.ml.math.Tensor
import hs.ml.math.TensorFactory

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

    fun split(parts: Int): Array<Node> {
        require(parts > 0 && data.col % parts == 0) { "Invalid parts: ${parts}" }

        val r = data.col / parts
        return Array(parts) { i ->
            val startCol = i * r
            Node(
                data.slice(startCol, startCol + r),
                children = listOf(this),
                operation = "split[$i/$parts]"
            ).also { node ->
                node._backward = {
                    for (row in 0 until grad.row)
                        for (col in 0 until node.grad.col)
                            grad[row, startCol + col] += node.grad[row, col]
                }
            }
        }
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