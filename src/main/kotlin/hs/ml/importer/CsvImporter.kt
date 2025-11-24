package hs.ml.importer

import hs.ml.math.Tensor
import java.io.File

class CsvImporter(val path: String): DataImporter {
    override fun read(): Pair<Tensor, Tensor> {
        val file = File(path)
        val lines = file.readLines()
        val data = lines.map { it.split(",").map {
            it.trim().toDouble()
        }}

        val row = data.size
        val col = data[0].size

        val x = Tensor(row, col - 1)
        val y = Tensor(row, 1)
        for (i in 0 until row) {
            for (j in 0 until col - 1) {
                x[i, j] = data[i][j]
            }
            y[i, 0] = data[i][col - 1]
        }

        return Pair(x, y)
    }

    override fun available(): Boolean {
        val file = File(path)
        return file.exists() && file.extension.lowercase() == "csv"
    }
}
