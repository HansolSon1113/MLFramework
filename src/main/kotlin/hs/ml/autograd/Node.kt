package hs.ml.autograd

import hs.ml.math.Tensor

class Node(var data: Tensor, val children: List<Node> = emptyList(), val debug: String = "") {
    var grad: Tensor = Tensor(data.row, data.col, 0.0)
    internal var _backward: () -> Unit = {}

    operator fun plus(other: Node): Node {
        val out = Node(this.data + other.data, listOf(this, other), "+")
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

    operator fun minus(other: Node): Node {
        return this + (-other)
    }

    operator fun unaryMinus(): Node {
        return this * -1.0
    }

    operator fun times(other: Node): Node {
        val out = Node(this.data * other.data, listOf(this, other), "*")

        out._backward = {
            this.grad = this.grad + (out.grad * other.data.T)
            other.grad = other.grad + (this.data.T * out.grad)
        }
        return out
    }

    operator fun times(scalar: Double): Node {
        val out = Node(this.data * scalar, listOf(this), "*$scalar")

        out._backward = {
            this.grad = this.grad + (out.grad * scalar)
        }
        return out
    }

    fun map(transform: (Double) -> Double, derivative: (Double) -> Double): Node {
        val outData = this.data.map(transform)
        val out = Node(outData, listOf(this), "map")

        out._backward = {
            val localGrad = this.data.map(derivative)
            this.grad = this.grad + (out.grad.hadamard(localGrad))
        }
        return out
    }

    fun pow(exponent: Int): Node {
        val out = Node(this.data.pow(exponent), listOf(this), "^$exponent")

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
        for(i in 0 until this.data.row) {
            for(j in 0 until this.data.col) {
                sum += this.data[i, j]
            }
        }
        val out = Node(Tensor(1, 1, sum / totalElements), listOf(this), "mean")

        out._backward = {
            val gradVal = out.grad[0, 0] / totalElements
            this.grad = this.grad + Tensor(this.data.row, this.data.col, gradVal)
        }
        return out
    }

    fun log(): Node {
        return this.map({ kotlin.math.ln(it) }, { 1.0 / it })
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

        if (initialGrad != null) {
            this.grad = initialGrad
        } else {
            this.grad = Tensor(data.row, data.col, 1.0)
        }

        for (node in topo.reversed()) {
            node._backward()
        }
    }
}