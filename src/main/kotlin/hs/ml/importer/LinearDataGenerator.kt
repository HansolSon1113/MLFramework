package hs.ml.importer

import hs.ml.data.DataBatch
import hs.ml.math.TensorFactory
import kotlin.random.Random

class LinearDataGenerator(
    val n: Int,
    val slope: Double,
    val bias: Double,
    val noise: Double
): DataImporter {
    override fun read(): DataBatch {
        val x = TensorFactory.create(n, 1) { i, j -> (i + 1).toDouble() }
        val y = TensorFactory.create(n, 1) { i, j ->
            slope * x[i, j] + bias + Random.nextDouble(-noise, noise)
        }

        return DataBatch(x, y)
    }

    override fun available(): Boolean = true
}
