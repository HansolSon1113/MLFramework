package hs.ml.model.nn.layer

import hs.ml.math.Tensor
import kotlin.random.Random

class Dense(val inputSize: Int, val outputSize: Int) : Layer() {
    val weights = Tensor(inputSize, outputSize)
    val bias = Tensor(1, outputSize)

    override fun forward(input: Tensor): Tensor {
        this.cache = input

        val weightedSum = input * weights

       val output = Tensor(weightedSum.row, weightedSum.col) { _: Int, _: Int -> Random.nextDouble(-0.1, 0.1) }
        for (i in 0 until weightedSum.row) {
            for (j in 0 until weightedSum.col) {
                output[i, j] = weightedSum[i, j] + bias[0, j]
            }
        }

        return output
    }

    override fun getParams(): List<Tensor> = listOf(weights, bias)
}