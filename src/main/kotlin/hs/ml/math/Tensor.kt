package hs.ml.math

class Tensor(val row: Int, val col: Int) {
    val data = MutableList(row) { MutableList(col) { 0.0 } }
    val T: Tensor
        get() = this.transpose()
    val shape: Pair<Int, Int>
        get() = Pair(row, col)

    constructor(row: Int, col: Int, value: Double): this(row, col) {
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                this[i, j] = value
    }

    constructor(row: Int, col: Int, value: (Int, Int) -> Double): this(row, col) {
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                this[i, j] = value(i, j)
    }

    operator fun get(idx: Int) = data[idx]
    operator fun get(i: Int, j: Int) = data[i][j]
    operator fun set(i: Int, j: Int, v: Double) { data[i][j] = v }

    operator fun unaryMinus(): Tensor {
        val tensor = Tensor(this.row, this.col)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                tensor[i, j] = -this[i, j]
        return tensor
    }

    operator fun plus(tensor: Tensor): Tensor {
        if (this.row != tensor.row || this.col != tensor.col)
            throw IllegalArgumentException("크기가 다른 두 행렬을 더할 수 없습니다.")

        val ans = Tensor(this.row, this.col)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                ans[i, j] = this[i][j] + tensor[i, j]

        return ans
    }

    operator fun plus(v: Double): Tensor {
        val ans = Tensor(this.row, this.col)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                ans[i, j] = this[i][j] + v

        return ans
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
            throw IllegalArgumentException("행렬 곱의 차원이 일치하지 않습니다.")

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

    fun transpose(): Tensor {
        val tensor = Tensor(this.col, this.row)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                tensor[j, i] = this[i, j]

        return tensor
    }

    fun max(): Double {
        var ans = Double.MIN_VALUE;
        for (i in 0 until this.row)
            for (j in 0 until this.col)
                if (ans < this[i, j])
                    ans = this[i, j]

        return ans
    }

    fun min(): Double {
        var ans = Double.MAX_VALUE;
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
