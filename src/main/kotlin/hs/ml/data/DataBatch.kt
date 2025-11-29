package hs.ml.data

import hs.ml.math.Tensor

data class DataBatch(
    val inputs: Tensor,
    val labels: Tensor
)
