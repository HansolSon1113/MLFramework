package hs.ml.train

import hs.ml.model.LinearRegressor
import hs.ml.model.LogisticRegressor
import hs.ml.model.Model

class ModelFactory {
    companion object {
        inline fun <reified T : Model> makeModel(): T {
            return when (T::class) {
                LinearRegressor::class -> LinearRegressor() as T
                LogisticRegressor::class -> LogisticRegressor() as T
                else -> throw IllegalArgumentException(
                    "Unsupported model type: ${T::class.simpleName}"
                )
            }
        }
    }
}