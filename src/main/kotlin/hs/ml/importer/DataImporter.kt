package hs.ml.importer

import hs.ml.math.Tensor

interface DataImporter {
    fun available(): Boolean
    fun read(): Pair<Tensor, Tensor>
}
