package hs.ml.model

import hs.ml.math.Tensor

abstract class Classifier: Model() {
    abstract fun classify(x: Tensor, threshold: Double = 0.5): Tensor
}
