package hs.ml.ui

import hs.ml.data.DataBatch
import hs.ml.data.DataPipeline
import hs.ml.importer.CsvImporter
import hs.ml.importer.DataImporter
import hs.ml.math.Tensor
import hs.ml.preprocessing.DataPreprocessor
import hs.ml.preprocessing.policy.MissingPolicy
import hs.ml.preprocessing.policy.ReplaceToAvgPolicy
import hs.ml.preprocessing.policy.ReplaceToZeroPolicy
import hs.ml.scaler.MinMaxScaler
import hs.ml.scaler.Scaler
import hs.ml.scaler.StandardScaler
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
    }
}
