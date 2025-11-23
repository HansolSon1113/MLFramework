package hs.ml.train

class Derivatives {
    companion object {
        val mse: (Double) -> Double = { diff ->
            2 * diff
        }
    }
}