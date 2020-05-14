cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(load-cnn)

find_package(Torch REQUIRED)

add_executable(load-cnn.ccp load-cnn.cpp)
target_link_libraries(load-cnn.ccp "${TORCH_LIBRARIES}")
set_property(TARGET load-cnn.ccp PROPERTY CXX_STANDARD 11)
