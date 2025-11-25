//src/ft_tokenizer.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "tokenizer_model.hpp"

namespace py = pybind11;


PYBIND11_MODULE(ft_tokenize, m) {
    m.doc() = "FT Tokenizer";

    py::class_<TokenizerModel>(m, "TokenizerModel")
        .def(py::init<>())  
        .def("train_from_textfile", &TokenizerModel::train_from_textfile)
        .def("save_model", &TokenizerModel::save_model)
        .def("load_model", &TokenizerModel::load_model)
        .def("encode_as_ids", &TokenizerModel::encode_as_ids)
        .def("encode_as_tokens", &TokenizerModel::encode_as_tokens)
        .def("decode_ids", &TokenizerModel::decode_ids)
        .def("decode_tokens", &TokenizerModel::decode_tokens)
        .def("token_to_id", &TokenizerModel::token_to_id)
        .def("id_to_token", &TokenizerModel::id_to_token)
        .def("get_token_size", &TokenizerModel::get_token_size)
        .def("get_vocab", &TokenizerModel::get_vocab);
}

