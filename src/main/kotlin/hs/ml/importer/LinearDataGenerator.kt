package hs.ml.importer

import hs.ml.data.DataBatch
import hs.ml.math.Tensor

class LinearDataGenerator(
    val n: Int,
    val slope: Double,
    val bias: Double,
    val noise: Double
): DataImporter {
    override fun read(): DataBatch {
        val x = Tensor(n, 1) { i, j -> (i + 1).toDouble() }
        val y = Tensor(n, 1) { i, j ->
            val xi = x[i, 0]
            slope * xi + bias + (Math.random() * 2 - 1) * noise
        }

        return DataBatch(x, y)
    }

    override fun available(): Boolean = true
}
