#CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(ft_tokenize LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)

add_subdirectory(extern/pybind11)

pybind11_add_module(ft_tokenize src/ft_tokenizer.cpp)

set_target_properties(ft_tokenize PROPERTIES
    OUTPUT_NAME "ft_tokenize"
)