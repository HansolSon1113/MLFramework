package hs.ml.loss

interface Loss {
    fun compute(yTrue: Double, yPred: Double): Double
}
