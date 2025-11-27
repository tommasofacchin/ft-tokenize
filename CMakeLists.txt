#CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(ft_tokenize LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")
message(STATUS "Python library: ${Python_LIBRARY}")

add_subdirectory(extern/pybind11)

pybind11_add_module(ft_tokenize src/ft_tokenizer.cpp)

target_link_libraries(ft_tokenize PRIVATE Python::Python)

set_target_properties(ft_tokenize PROPERTIES
    OUTPUT_NAME "ft_tokenize"
)

