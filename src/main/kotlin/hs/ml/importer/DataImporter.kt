package hs.ml.importer

import hs.ml.data.DataBatch

interface DataImporter {
    fun available(): Boolean
    fun read(): DataBatch
}
