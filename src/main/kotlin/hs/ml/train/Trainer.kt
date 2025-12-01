package hs.ml.train

import hs.ml.data.DataBatch
import hs.ml.math.Tensor
import hs.ml.model.Model

class Trainer(
    val model: Model,
) {
    fun trainStep(batch: DataBatch): Double {
        // Forward pass
        val predictions = model.forward(batch.inputs)

        // Compute loss and gradients
        val loss = model.param.loss.compute(batch.labels, predictions)
        val gradients = model.param.loss.gradient(batch.labels, predictions)

        // Update model parameters
        val (w, b) = model.param.optimizer.step(Pair(model.weights, model.bias), gradients)
        model.weights = w
        model.bias = b

        return loss
    }

    fun train(batch: DataBatch, epochs: Int = 1000, verbose: Boolean = false) {
        model.weights = Tensor(batch.inputs.col, batch.labels.col) { i, j ->
            Math.random() * 0.01
        }
        model.bias = 0.0

        for (epoch in 1..epochs) {
            val loss = trainStep(batch)
            if (verbose && epoch % 100 == 0)
                println("Epoch $epoch, Loss: $loss")
        }
    }
}
