package hs.ml.util

import hs.ml.data.DataBatch
import hs.ml.math.Tensor
import hs.ml.math.TensorFactory
import kotlin.math.roundToInt

fun formatBytes(bytes: Long): String {
    val units = listOf("B", "KB", "MB", "GB", "TB")
    var idx = 0
    var b = bytes.toDouble();
    while (b >= 1024) {
        b /= 1024
        idx += 1
    }

    return "${(b * 100).roundToInt().toDouble() / 100} ${units[idx]}"
}

fun trainTestSplit(data: DataBatch, trainRatio: Double): Pair<DataBatch, DataBatch> {
    val indices = (0 ..< data.inputs.row).toList().shuffled()
    val trainSize = (data.inputs.row * trainRatio).toInt()
    val trainIndices = indices.take(trainSize)
    val testIndices = indices.drop(trainSize)

    fun createBatch(batchIndices: List<Int>) = DataBatch(
        inputs = TensorFactory.create(batchIndices.size, data.inputs.col) { i, j ->
            data.inputs[batchIndices[i], j]
        },
        labels = TensorFactory.create(batchIndices.size, data.labels.col) { i, j ->
            data.labels[batchIndices[i], j]
        }
    )

    return Pair(createBatch(trainIndices), createBatch(testIndices))
}