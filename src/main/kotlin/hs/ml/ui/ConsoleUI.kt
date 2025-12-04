package hs.ml.ui

import hs.ml.data.DataBatch
import hs.ml.importer.CsvImporter
import hs.ml.math.Tensor
import java.io.File
import java.util.Scanner

class ConsoleUI {
    private val scanner = Scanner(System.`in`)

    private var data = DataBatch(Tensor(), Tensor())

    fun getData(): DataBatch {
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

        return CsvImporter(filePath).read()
    }

    fun start() {
        println("OOP Machine Learning Project")

        println("\n**데이터 전처리 단계**")
        data = getData()
    }
}
