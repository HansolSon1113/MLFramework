plugins {
    kotlin("jvm") version "2.2.20"
}

group = "hs.ml"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(23)
}

tasks.register<Exec>("buildMetal") {
    group = "build"
    description = "Build Metal GPU acceleration native library"

    commandLine("bash", "${project.rootDir}/scripts/build-metal.sh")

    onlyIf {
        System.getProperty("os.name").lowercase().contains("mac")
    }
}

tasks.jar {
    from(configurations.runtimeClasspath.get().map { if (it.isDirectory) it else zipTree(it) })

    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}