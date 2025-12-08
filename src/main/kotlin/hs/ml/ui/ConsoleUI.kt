package hs.ml.ui

import hs.ml.autograd.Node
import hs.ml.data.DataBatch
import hs.ml.data.DataPipeline
import hs.ml.importer.CsvImporter
import hs.ml.importer.DataImporter
import hs.ml.loss.BinaryCrossEntropy
import hs.ml.loss.MeanSquaredError
import hs.ml.math.Tensor
import hs.ml.metric.F1
import hs.ml.metric.RootMeanSquaredError
import hs.ml.model.Model
import hs.ml.model.classifier.LogisticRegressor
import hs.ml.model.regressor.LinearRegressor
import hs.ml.preprocessing.DataPreprocessor
import hs.ml.preprocessing.policy.MissingPolicy
import hs.ml.preprocessing.policy.ReplaceToAvgPolicy
import hs.ml.preprocessing.policy.ReplaceToZeroPolicy
import hs.ml.scaler.MinMaxScaler
import hs.ml.scaler.Scaler
import hs.ml.scaler.StandardScaler
import hs.ml.train.Trainer
import hs.ml.train.optimizer.Adam
import hs.ml.train.optimizer.SGD
import hs.ml.util.trainTestSplit
import java.io.File
import java.util.Scanner

class ConsoleUI {
    private val scanner = Scanner(System.`in`)

    private var data = DataBatch(Tensor(), Tensor())

    private fun select(options: List<String>): Int {
        for ((index, option) in options.withIndex()) {
            println("${index + 1}. $option")
        }

        var choice: Int
        do {
            print("> ")
            val input = scanner.nextLine()
            choice = input.toIntOrNull() ?: -1
            if (choice !in 1..options.size) {
                println("유효한 선택지가 아닙니다. 다시 시도해주세요.")
            } else {
                break
            }
        } while (true)
        return choice
    }

    fun getImporter(): DataImporter {
        var filePath: String
        do {
            print("CSV 파일 경로 : ")
            filePath = scanner.nextLine()
            val file = File(filePath)
            if (!file.exists() || file.extension.lowercase() != "csv") {
                println("유효한 CSV 파일 경로가 아닙니다. 다시 시도해주세요.")
                continue
            } else {
                break
            }
        } while (true)

        return CsvImporter(filePath)
    }

    fun getScaler(): Scaler? {
        println("스케일러 선택:")
        val s = select(listOf("StandardScaler", "MinMaxScaler", "None"))
        return when (s) {
            1 -> StandardScaler()
            2 -> MinMaxScaler()
            else -> null
        }
    }

    fun getMissingPolicy(): MissingPolicy {
        println("결측치 처리 정책 선택:")
        val p = select(listOf("평균값으로 대체", "0으로 대체"))

        return when (p) {
            1 -> ReplaceToAvgPolicy()
            else -> ReplaceToZeroPolicy()
        }
    }

    fun getModel(): Model {
        println("모델 선택:")
        val m = select(listOf("Linear Regression", "Logistic Regression", "Neural Network"))

        val model = when (m) {
            1 -> LinearRegressor()
            2 -> LogisticRegressor()
            3 -> TODO("추후 구현")
            else -> throw IllegalStateException("유효하지 않은 모델 선택")
        }

        println("loss 함수 선택:")
        val l = select(listOf("Mean Squared Error", "Binary Cross Entropy"))
        model.param.loss = when (l) {
            1 -> MeanSquaredError()
            else -> BinaryCrossEntropy()
        }

        println("optimizer 선택:")
        val o = select(listOf("SGD", "Adam"))
        model.param.optimizer = when (o) {
            1 -> SGD()
            else -> Adam()
        }

        println("Metric 선택 (여러개 선택 가능):")
        val metrics = mutableListOf<String>()
        val availableMetrics = listOf("Root Mean Squared Error", "F1")
        while (true) {
            val metricOptions = availableMetrics.filter { it !in metrics }
            if (metricOptions.isEmpty()) {
                println("선택 가능한 Metric이 더 이상 없습니다.")
                break
            }

            val mt = select(metricOptions + listOf("완료"))
            if (mt == metricOptions.size + 1) {
                break
            } else {
                metrics.add(metricOptions[mt - 1])
            }
        }

        for (metric in metrics) {
            when (metric) {
                "Root Mean Squared Error" -> model.param.metric.add(RootMeanSquaredError())
                else -> model.param.metric.add(F1())
            }
        }

        return model
    }

    fun start() {
        println("OOP Machine Learning Project")

        println("\n**데이터 전처리 단계**")
        val importer = getImporter()
        val scaler = getScaler()
        val policy = getMissingPolicy()

        val pipeline = DataPipeline(
            importer = importer,
            preprocessor = DataPreprocessor(policy, scaler)
        )

        data = pipeline.run()
        println("데이터 전처리 완료. 데이터 크기: ${data.inputs.shape}")

        println("\n전처리된 데이터 미리보기 (첫 5개 행):")
        val previewRows = minOf(5, data.inputs.row)
        for (i in 0 until previewRows) {
            val rowValues = (0 until data.inputs.col).joinToString(", ") { j ->
                String.format("%.4f", data.inputs[i, j])
            }

            println("[$i] $rowValues")
        }

        val (train, test) = trainTestSplit(data, 0.8)

        println("\n**모델 생성 단계**")
        val model = getModel()
        println("모델 생성 완료: $model")

        println("\n**학습 시작**")
        val trainer = Trainer(model)
        print("epoch = ")
        val epochs = scanner.nextLine().toIntOrNull() ?: 1000
        trainer.train(train, epochs = epochs, verbose = true)

        println("\n**학습 완료**")
        println("Model = $model")

        println("\n**테스트 데이터 평가**")
        for (i in 0 until 5) {
            val yPred = model.forward(Node(test.inputs))
            val prediction = yPred.data[i, 0]
            val actual = test.labels[i, 0]
            val diff = prediction - actual

            println(
                "[$i] 실제값: ${String.format("%.4f", actual)} | 예측값: ${
                    String.format(
                        "%.4f",
                        prediction
                    )
                } | 오차: ${String.format("%.4f", diff)}"
            )
        }
    }
}
