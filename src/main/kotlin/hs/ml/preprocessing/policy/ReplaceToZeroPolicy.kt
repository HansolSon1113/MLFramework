package hs.ml.preprocessing.policy

import hs.ml.math.Tensor

class ReplaceToZeroPolicy: MissingPolicy {
    override fun handle(tensor: Tensor): Tensor {
        println("MissingPolicy: 결측치를 0으로 대체합니다.")
        val data = tensor.data.map { row->
            row.map{ value-> if (value.isNaN()) 0.0 else value }.toDoubleArray()
        }.toTypedArray()

        return Tensor(data)
    }
}