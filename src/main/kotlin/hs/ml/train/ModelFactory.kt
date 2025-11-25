package hs.ml.train

import hs.ml.loss.Loss
import hs.ml.metric.Metric
import hs.ml.model.LinearRegressor
import hs.ml.model.LogisticRegressor
import hs.ml.model.Model
import hs.ml.model.ModelParameter
import hs.ml.scaler.Scaler

class ModelFactory<T: Model>(
    private val model: T
) {
    private val param: ModelParameter = ModelParameter()

    companion object {
        inline fun <reified T : Model> create(): ModelFactory<T> {
            val model = when (T::class) {
                LinearRegressor::class -> LinearRegressor()
                LogisticRegressor::class -> LogisticRegressor()
                else -> throw IllegalArgumentException(
                    "Unsupported model type: ${T::class.simpleName}"
                )
            } as T

            return ModelFactory(model)
        }
    }

    fun setScaler(scaler: Scaler): ModelFactory<T> {
        param.scaler = scaler
        return this
    }

    fun setLoss(loss: Loss): ModelFactory<T> {
        param.loss = loss
        return this
    }

    fun addMetric(metric: Metric): ModelFactory<T> {
        param.metric.add(metric)
        return this
    }

    fun getModel(): T {
        model.param = param
        return model
    }
}
