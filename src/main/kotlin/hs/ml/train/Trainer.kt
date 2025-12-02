package hs.ml.train

import hs.ml.autograd.Node
import hs.ml.data.DataBatch
import hs.ml.math.Tensor
import hs.ml.metric.Metric
import hs.ml.model.Model

class Trainer(
    val model: Model,
) {
    fun trainStep(batch: DataBatch): Double {
        val inputs = Node(batch.inputs)

        // Forward pass
        val pred: Node = model.forward(inputs)

        // Compute loss and gradients
        val loss = model.param.loss.compute(batch.labels, pred.data)
        val (grad, _) = model.param.loss.gradient(batch.labels, pred.data)

        // Update model parameters
        pred.backward(grad)
        model.param.optimizer.step(model.params())

        model.params().forEach {
            it.grad = Tensor(it.data.row, it.data.col, 0.0)
        }

        return loss
    }

    fun evaluate(batch: DataBatch, vararg metrics: Metric): Map<String, Double> {
        val inputs = Node(batch.inputs)

        val pred = model.forward(inputs)

        val results = mutableMapOf<String, Double>()

        val loss = model.param.loss.compute(batch.labels, pred.data)
        results["Loss"] = loss

        val targetMetrics = if (metrics.isEmpty()) model.param.metric else metrics.toList()

        targetMetrics.forEach { metric ->
            val metricName = metric::class.simpleName ?: "Metric"
            val score = metric.evaluate(batch.labels, pred.data)
            results[metricName] = score
        }

        return results
    }

    fun train(batch: DataBatch, epochs: Int = 1000, verbose: Boolean = false, showMetric: Int = 100) {
        val startEpoch = model.epoch + 1
        val targetEpoch = model.epoch + epochs

        if (verbose) {
            println("Training started: Epoch $startEpoch to $targetEpoch")
        }

        for (i in startEpoch..targetEpoch) {
            trainStep(batch)
            model.epoch = i

            if (verbose && i % showMetric == 0) {
                val metrics = evaluate(batch)
                val logMsg = metrics.entries.joinToString(", ") { (name, value) ->
                    "$name: ${String.format("%.4f", value)}"
                }
                println("Epoch $i | $logMsg")
            }
        }
    }
}
