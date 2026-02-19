package hs.ml.math

import kotlin.math.pow

abstract class Tensor(val row: Int, val col: Int) {
    enum class Axis {
        VERTICAL,
        HORIZONTAL
    }

    abstract val data: MutableList<MutableList<Double>>

    val T: Tensor
        get() = this.transpose()
    val shape: Pair<Int, Int>
        get() = Pair(row, col)

    operator fun get(idx: Int) = data[idx]
    operator fun get(i: Int, j: Int) = data[i][j]
    operator fun set(i: Int, j: Int, v: Double) {
        data[i][j] = v
    }

    abstract operator fun unaryMinus(): Tensor
    abstract operator fun plus(tensor: Tensor): Tensor
    abstract operator fun minus(tensor: Tensor): Tensor
    abstract operator fun times(tensor: Tensor): Tensor
    abstract operator fun times(scalar: Double): Tensor
    abstract infix fun hadamard(tensor: Tensor): Tensor
    abstract fun transpose(): Tensor

    fun map(transform: (Double) -> Double): Tensor {
        return createTensor(this.row, this.col) { i, j ->
            transform(this[i, j])
        }
    }

    fun pow(exponent: Int): Tensor {
        return this.map { it.pow(exponent) }
    }

    fun sum(axis: Axis): Tensor {
        return when (axis) {
            Axis.VERTICAL -> createTensor(1, this.col) { _, j -> (0..<row).sumOf { i -> this[i, j] } }
            Axis.HORIZONTAL -> createTensor(this.row, 1) { i, _ -> this[i].sum() }
        }
    }

    fun max(): Double {
        var ans = Double.NEGATIVE_INFINITY
        for (i in 0 until this.row)
            for (j in 0 until this.col)
                if (ans < this[i, j])
                    ans = this[i, j]

        return ans
    }

    fun min(): Double {
        var ans = Double.POSITIVE_INFINITY
        for (i in 0 until this.row)
            for (j in 0 until this.col)
                if (ans > this[i, j])
                    ans = this[i, j]

        return ans
    }

    fun slice(startCol: Int, endCol: Int): Tensor {
        require(startCol >= 0 && endCol <= this.col && startCol < endCol) {
            "Invalid slice range: [$startCol, $endCol) for tensor with ${this.col} columns"
        }
        return createTensor(this.row, endCol - startCol) { i, j ->
            this[i, startCol + j]
        }
    }

    override fun toString(): String {
        val builder = StringBuilder()
        for (i in 0..<this.row) {
            for (j in 0..<this.col)
                builder.append("${this[i, j]},\t")
            builder.append("\n")
        }

        return builder.toString()
    }

    protected abstract fun createTensor(row: Int, col: Int, init: (Int, Int) -> Double = { _, _ -> 0.0 }): Tensor
}
