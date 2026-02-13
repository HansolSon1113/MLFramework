package hs.ml.math

import hs.ml.math.metal.MetalConfig
import hs.ml.math.metal.MetalTensor

object TensorFactory {
    var useGpu: Boolean = false

    fun create(row: Int, col: Int): Tensor {
        return if (useGpu && MetalConfig.isAvailable()) {
            MetalTensor(row, col)
        } else {
            CpuTensor(row, col)
        }
    }

    fun create(row: Int, col: Int, value: Double): Tensor {
        return if (useGpu && MetalConfig.isAvailable()) {
            MetalTensor(row, col, value)
        } else {
            CpuTensor(row, col, value)
        }
    }

    fun create(row: Int, col: Int, init: (Int, Int) -> Double): Tensor {
        return if (useGpu && MetalConfig.isAvailable()) {
            MetalTensor(row, col, init)
        } else {
            CpuTensor(row, col, init)
        }
    }

    fun create(data: MutableList<MutableList<Double>>): Tensor {
        return if (useGpu && MetalConfig.isAvailable()) {
            MetalTensor(data)
        } else {
            CpuTensor(data)
        }
    }

    fun create(dataArray: Array<DoubleArray>): Tensor {
        return if (useGpu && MetalConfig.isAvailable()) {
            MetalTensor(dataArray)
        } else {
            CpuTensor(dataArray)
        }
    }
}

