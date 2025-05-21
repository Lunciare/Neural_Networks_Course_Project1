cmake_minimum_required(VERSION 3.10)
project(neural_net)

set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/external/eigen)
include_directories(${PROJECT_SOURCE_DIR}/external/eigenRand)
include_directories(${PROJECT_SOURCE_DIR}/src)

set(SOURCES
    src/main.cpp
    src/ActivationFunctions/ActivationFunction.cpp
    src/Optimizer/Optimizer.cpp
    src/LossFunctions/LossFunction.cpp
    src/Loader/MNISTLoader.cpp
    src/Layers/Layer.cpp
    src/Model/Model.cpp
    src/Tests/Tests.cpp
)

add_executable(neural_net
    src/main.cpp
    src/ActivationFunctions/ActivationFunction.cpp
    src/Optimizer/Optimizer.cpp
    src/LossFunctions/LossFunction.cpp
    src/Loader/MNISTLoader.cpp
    src/Layers/Layer.cpp
    src/Model/Model.cpp
    src/Tests/Tests.cpp
)