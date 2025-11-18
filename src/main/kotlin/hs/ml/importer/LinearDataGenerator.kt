package hs.ml.importer

import hs.ml.data.Tensor

class LinearDataGenerator(
    val n: Int,
    val slope: Double,
    val bias: Double,
    val noise: Double
): DataImporter {
    override fun read(): Pair<Tensor, Tensor> {
        val x = Tensor(n, 1) { i, j -> (i + 1).toDouble() }
        val y = Tensor(n, 1) { i, j ->
            val xi = x[i, 0]
            slope * xi + bias + (Math.random() * 2 - 1) * noise
        }

        return Pair(x, y)
    }

    override fun available(): Boolean = true
}
