cmake_minimum_required(VERSION 3.24)
project(pidnet_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLX
find_package(MLX CONFIG REQUIRED)

# Find SentencePiece (optional, for tokenization)
find_package(PkgConfig)
pkg_check_modules(SENTENCEPIECE sentencepiece)

# PID-Net inference engine
add_executable(pidnet
    pidnet_engine.cpp
)

target_link_libraries(pidnet PRIVATE mlx)
target_include_directories(pidnet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(SENTENCEPIECE_FOUND)
    target_link_libraries(pidnet PRIVATE ${SENTENCEPIECE_LIBRARIES})
    target_include_directories(pidnet PRIVATE ${SENTENCEPIECE_INCLUDE_DIRS})
    target_compile_definitions(pidnet PRIVATE HAS_SENTENCEPIECE=1)
endif()

# Optimization flags for Apple Silicon
if(APPLE)
    target_compile_options(pidnet PRIVATE -O2 -mcpu=apple-m2 -fno-fast-math)
endif()
