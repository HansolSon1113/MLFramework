package hs.ml.preprocessing

import hs.ml.data.DataBatch
import hs.ml.preprocessing.policy.MissingPolicy
import hs.ml.scaler.Scaler

class DataPreprocessor(
    private val missingPolicy: MissingPolicy,
    private val scaler: Scaler?
){
    fun process(batch: DataBatch):DataBatch{
        println("DataPreprocessor: 전처리 시작...")
        val processedInputs = missingPolicy.handle(batch.inputs)

        if (scaler == null) {
            return DataBatch(
                inputs = processedInputs,
                labels = batch.labels
            )
        }

        scaler.fit(processedInputs)
        val scaledInputs = scaler.transform(processedInputs)

        val processedBatch = DataBatch(
            inputs = scaledInputs,
            labels = batch.labels
        )
        println("DataPreprocessor: 전처리 완료.")
        return processedBatch
    }
}
