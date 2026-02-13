package hs.ml.math

class CpuTensor : Tensor {
    override val data: MutableList<MutableList<Double>>

    constructor(row: Int, col: Int) : super(row, col) {
        data = MutableList(row) { MutableList(col) { 0.0 } }
    }

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
        for (i in 0 until this.row) {
            for (j in 0 until this.col) {
                this.data[i][j] = dataArray[i][j]
            }
        }
    }

    override fun unaryMinus(): Tensor {
        val tensor = CpuTensor(this.row, this.col)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                tensor[i, j] = -this[i, j]
        return tensor
    }

    override fun plus(tensor: Tensor): Tensor {
        if (this.row == tensor.row && this.col == tensor.col) {
            val ans = CpuTensor(this.row, this.col)
            for (i in 0..<this.row)
                for (j in 0..<this.col)
                    ans[i, j] = this[i, j] + tensor[i, j]
            return ans
        } else if (tensor.row == 1 && this.col == tensor.col) {
            val ans = CpuTensor(this.row, this.col)
            for (i in 0..<this.row) {
                for (j in 0..<this.col) {
                    ans[i, j] = this[i, j] + tensor[0, j]
                }
            }
            return ans
        } else {
            throw IllegalArgumentException("Incompatible tensor shapes for addition")
        }
    }

    override fun minus(tensor: Tensor): Tensor {
        if (this.row != tensor.row || this.col != tensor.col)
            throw IllegalArgumentException("Incompatible tensor shapes for subtraction")

        val ans = CpuTensor(this.row, this.col)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                ans[i, j] = this[i, j] - tensor[i, j]

        return ans
    }

    override fun times(tensor: Tensor): Tensor {
        if (this.col != tensor.row)
            throw IllegalArgumentException("Matrix dimensions incompatible: (${this.row}, ${this.col}) x (${tensor.row}, ${tensor.col})")

        val ans = CpuTensor(this.row, tensor.col)
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

    override fun times(scalar: Double): Tensor {
        return CpuTensor(this.row, this.col) { i, j ->
            this[i, j] * scalar
        }
    }

    override fun hadamard(tensor: Tensor): Tensor {
        if (this.row != tensor.row || this.col != tensor.col)
            throw IllegalArgumentException("Shapes must match for Hadamard product")
        return CpuTensor(this.row, this.col) { i, j ->
            this[i, j] * tensor[i, j]
        }
    }

    override fun transpose(): Tensor {
        val tensor = CpuTensor(this.col, this.row)
        for (i in 0..<this.row)
            for (j in 0..<this.col)
                tensor[j, i] = this[i, j]

        return tensor
    }

    override fun createTensor(row: Int, col: Int, init: (Int, Int) -> Double): Tensor {
        return CpuTensor(row, col, init)
    }
}
