package hs.ml

import hs.ml.data.DataPipeline
import hs.ml.importer.CsvImporter
import hs.ml.loss.MeanSquaredError
import hs.ml.metric.RootMeanSquaredError
import hs.ml.model.LinearRegressor
import hs.ml.preprocessing.DataPreprocessor
import hs.ml.preprocessing.policy.ReplaceToAvgPolicy
import hs.ml.scaler.StandardScaler
import hs.ml.train.ModelFactory
import hs.ml.train.Trainer
import hs.ml.train.optimizer.Adam
import hs.ml.util.formatBytes
import java.io.File

fun main() {
    println("\n\n================================")
    println("OOP Machine Learning Project")
    println("PWD : ${File(".").canonicalFile}")
    println("CPU : ${Runtime.getRuntime().availableProcessors()} cores")
    println("Mem : ${formatBytes(Runtime.getRuntime().maxMemory())}")
    println("================================\n\n")
    println()

    println("**데이터 불러오기 및 전처리 단계**\n")
    println("데이터 불러오는 중...")

    val importer = CsvImporter("data/housing.csv")
    val pipeline = DataPipeline(
        importer = importer,
        preprocessor = DataPreprocessor(
            missingPolicy = ReplaceToAvgPolicy(),
            scaler = StandardScaler()
        )
    )
    val batch = pipeline.run()

    println("데이터 불러오기 완료!")
    println("x: ${batch.inputs.shape}, y: ${batch.labels.shape}")

    println("\n**모델 선택 단계**\n")

    val model = ModelFactory.create<LinearRegressor>()
        .setLoss(MeanSquaredError())
        .setOptimizer(Adam(lr = 0.0001))
        .addMetric(RootMeanSquaredError())
        .getModel()

    println("모델 생성 완료! : $model")

    println("\n**학습 시작**\n")
    val trainer = Trainer(model)
    trainer.train(batch, epochs = 1000, verbose = true)

    println("\n**학습 완료**\n")
    println(model)

    println("\n예측 결과 예시:")
    val sampleInput = batch.inputs
    val predictions = model.forward(sampleInput)
    for (i in 0 until 5) {
        println("입력: ${sampleInput.data[i]} => 예측: ${predictions.data[i]}, 실제값: ${batch.labels.data[i]}")
    }
}
