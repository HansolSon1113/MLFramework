package hs.ml.math

import kotlin.math.pow

class Tensor(val row: Int, val col: Int) {
    enum class Axis {
        VERTICAL,
        HORIZONTAL
    }

    val data = MutableList(row) { MutableList(col) { 0.0 } }
    val T: Tensor
        get() = this.transpose()
    val shape: Pair<Int, Int>
        get() = Pair(row, col)

    constructor(): this(0, 0)

    constructor(row: Int, col: Int, value: Double) : this(row, col) {
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                this[i, j] = value
    }

    constructor(row: Int, col: Int, value: (Int, Int) -> Double) : this(row, col) {
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                this[i, j] = value(i, j)
    }

    constructor(data: MutableList<MutableList<Double>>) : this(data.size, data[0].size) {
        for (i in 0..<this.row) {
            for (j in 0..<this.col) {
                this.data[i][j] = data[i][j]
            }
        }
    }

    constructor(dataArray: Array<DoubleArray>) : this(dataArray.size, dataArray[0].size) {
        // Array 타입을 MutableList로 변환하는 로직을 내부에서 처리합니다.
        for (i in 0 until this.row) {
            for (j in 0 until this.col) {
                this.data[i][j] = dataArray[i][j]
            }
        }
    }

    operator fun get(idx: Int) = data[idx]
    operator fun get(i: Int, j: Int) = data[i][j]
    operator fun set(i: Int, j: Int, v: Double) {
        data[i][j] = v
    }

    operator fun unaryMinus(): Tensor {
        val tensor = Tensor(this.row, this.col)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                tensor[i, j] = -this[i, j]
        return tensor
    }

    operator fun plus(tensor: Tensor): Tensor {
        if (this.row == tensor.row && this.col == tensor.col) {
            val ans = Tensor(this.row, this.col)
            for (i in 0..<this.row)
                for (j in 0..<this.col)
                    ans[i, j] = this[i, j] + tensor[i, j]
            return ans
        }
        //Broadcasting
        else if (tensor.row == 1 && this.col == tensor.col) {
            val ans = Tensor(this.row, this.col)
            for (i in 0..<this.row) {
                for (j in 0..<this.col) {
                    ans[i, j] = this[i, j] + tensor[0, j] // 같은 열의 값을 계속 더함
                }
            }
            return ans
        } else {
            throw IllegalArgumentException("더할 수 있는 형태가 아닙니다.")
        }
    }

    operator fun minus(tensor: Tensor): Tensor {
        if (this.row != tensor.row || this.col != tensor.col)
            throw IllegalArgumentException("크기가 다른 두 행렬을 더할 수 없습니다.")

        val ans = Tensor(this.row, this.col)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                ans[i, j] = this[i, j] - tensor[i, j]

        return ans
    }

    operator fun times(tensor: Tensor): Tensor {
        if (this.col != tensor.row)
            throw IllegalArgumentException("행렬 곱의 차원이 일치하지 않습니다. (${this.row}, ${this.col}) x (${tensor.row}, ${tensor.col})")

        val ans = Tensor(this.row, tensor.col)
        for (i in 0 until this.row) {
            for (j in 0 until tensor.col) {
                var sum = 0.0
                for (k in 0 until this.col)
                    sum += this[i, k] * tensor[k, j]
                ans[i, j] = sum
            }
        }
        return ans
    }

    operator fun times(scalar: Double): Tensor {
        return Tensor(this.row, this.col) { i, j ->
            this[i, j] * scalar
        }
    }

    infix fun hadamard(tensor: Tensor): Tensor {
        if (this.row != tensor.row || this.col != tensor.col)
            throw IllegalArgumentException("Shapes must match for Hadamard product")
        return Tensor(this.row, this.col) { i, j ->
            this[i, j] * tensor[i, j]
        }
    }

    fun map(transform: (Double) -> Double): Tensor {
        return Tensor(this.row, this.col) { i, j ->
            transform(this[i, j])
        }
    }

    //Element-wise pow
    fun pow(exponent: Int): Tensor {
        return this.map { it.pow(exponent) }
    }

    fun transpose(): Tensor {
        val tensor = Tensor(this.col, this.row)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                tensor[j, i] = this[i, j]

        return tensor
    }

    fun sum(axis: Axis): Tensor {
        return when (axis) {
            Axis.VERTICAL -> Tensor(1, this.col) { _, j -> (0 ..< row).sumOf { i -> this[i, j] } }
            Axis.HORIZONTAL -> Tensor(this.row, 1) { i, _ -> this[i].sum() }
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

    override fun toString(): String {
        val builder = StringBuilder()
        for (i in 0..<this.row) {
            for (j in 0..<this.col)
                builder.append("${this[i, j]},\t")
            builder.append("\n")
        }

        return builder.toString()
    }
}
