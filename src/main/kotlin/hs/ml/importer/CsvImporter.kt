package hs.ml.importer

import hs.ml.data.Tensor

class CsvImporter(val path: String): DataImporter {
    override fun read(): Pair<Tensor, Tensor> {
        TODO("Not yet implemented")
    }

    override fun available(): Boolean = true
}