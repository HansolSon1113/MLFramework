package hs.ml.preprocessing.policy

import hs.ml.math.Tensor

class ReplaceToAvgPolicy : MissingPolicy {
    override fun handle(tensor: Tensor): Tensor {
        println("MissingPolicy: 결측치를 각 피처(열)의 평균값으로 대체합니다.")
        val columnMeans = DoubleArray(tensor.col) { colIdx ->
            var sum = 0.0
            var count = 0

            for (rowIdx in 0 until tensor.row) {
                val value = tensor[rowIdx, colIdx]
                if (!value.isNaN()) {
                    sum += value
                    count++
                }
            }

            if (count == 0) 0.0 else sum / count
        }

        val newData = MutableList(tensor.row) { rowIdx ->
            MutableList(tensor.col) { colIdx ->
                val originalValue = tensor[rowIdx, colIdx]

                if (originalValue.isNaN()) {
                    columnMeans[colIdx]
                } else {
                    originalValue
                }
            }
        }

        return Tensor(newData)
    }
}