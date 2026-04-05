package hs.ml.math

import hs.ml.util.EPSILON
import kotlin.math.pow
import kotlin.math.sqrt

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

    fun slice(start: Int, end: Int, axis: Axis): Tensor {
        return when (axis) {
            Axis.VERTICAL -> {
                require(start >= 0 && end <= this.col && start < end) {
                    "Invalid slice range: [$start, $end] for tensor with ${this.col} columns"
                }
                createTensor(this.row, end - start) { i, j ->
                    this[i, start + j]
                }
            }

            Axis.HORIZONTAL -> {
                require(start >= 0 && end <= this.row && start < end) {
                    "Invalid slice range: [$start, $end] for tensor with ${this.col} rows"
                }
                createTensor(end - start, this.col) { i, j ->
                    this[start + i, j]
                }
            }
        }
    }

    fun concat(other: Tensor, axis: Axis): Tensor {
        return when (axis) {
            Axis.VERTICAL -> {
                require(this.row == other.row) {
                    "Row dimensions must match for concatenation: ${this.row} != ${other.row}"
                }
                createTensor(this.row, this.col + other.col) { i, j ->
                    if (j < this.col) this[i, j] else other[i, j - this.col]
                }
            }

            Axis.HORIZONTAL -> {
                require(this.col == other.col) {
                    "Col dimensions must match for concatenation: ${this.col} != ${other.col}"
                }
                createTensor(this.row + other.row, this.col) { i, j ->
                    if (i < this.row) this[i, j] else other[i - this.row, j]
                }
            }
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
