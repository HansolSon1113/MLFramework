package hs.ml.data

class Tensor(val row: Int, val col: Int) {
    val data: MutableList<MutableList<Double>> = MutableList(row) {
        MutableList(col) { 0.0 }
    }

    constructor(row: Int, col: Int, value: Double): this(row, col) {
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                this[i][j] = value
    }

    operator fun get(idx: Int) = data[idx]

    operator fun unaryMinus(): Tensor {
        val tensor = Tensor(this.row, this.col)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                tensor[i][j] = -this[i][j]
        return tensor
    }

    operator fun plus(tensor: Tensor): Tensor {
        if (this.row != tensor.row || this.col != tensor.col)
            throw IllegalArgumentException("크기가 다른 두 행렬을 더할 수 없습니다.")

        val ans = Tensor(this.row, this.col)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                ans[i][j] = this[i][j] + tensor[i][j]

        return ans
    }

    operator fun minus(tensor: Tensor): Tensor {
        if (this.row != tensor.row || this.col != tensor.col)
            throw IllegalArgumentException("크기가 다른 두 행렬을 더할 수 없습니다.")

        val ans = Tensor(this.row, this.col)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                ans[i][j] = this[i][j] - tensor[i][j]

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
                    sum += this[i][k] * tensor[k][j]
                ans[i][j] = sum
            }
        }
        return ans
    }

    override fun toString(): String {
        val builder = StringBuilder()
        for (i in 0..<this.row) {
            for (j in 0..<this.col)
                builder.append("${this[i][j]},\t")
            builder.append("\n")
        }

        return builder.toString()
    }
}
