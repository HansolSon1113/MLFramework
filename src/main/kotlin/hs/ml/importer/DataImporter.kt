package hs.ml.importer

import hs.ml.data.Tensor

interface DataImporter {
    fun available(): Boolean
    fun read(): Pair<Tensor, Tensor>
}
