package hs.ml.preprocessing.policy

import hs.ml.math.Tensor
import hs.ml.math.TensorFactory

class ReplaceToZeroPolicy : MissingPolicy {
    override fun handle(tensor: Tensor): Tensor {
        val data = tensor.data.map { row ->
            row.map { value -> if (value.isNaN()) 0.0 else value }.toDoubleArray()
        }.toTypedArray()

        return TensorFactory.create(data)
    }
}