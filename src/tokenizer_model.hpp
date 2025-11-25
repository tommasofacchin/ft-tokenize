//src/tokenizer_model.hpp
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

enum class TokenizerMode {
    WORD,
    BPE
};

class TokenizerModel {
public:
    TokenizerModel();

    void train_from_textfile(const std::string &input_file,
                            size_t vocab_size = 10000,
                            const std::vector<std::string> &user_defined_symbols = {},
                            TokenizerMode mode = TokenizerMode::WORD);

    void train_word_level(const std::string &input_file,
                        size_t vocab_size,
                        const std::vector<std::string> &user_defined_symbols);

    void train_bpe(const std::string &input_file,
                size_t vocab_size,
                const std::vector<std::string> &user_defined_symbols);
                               
    void save_model(const std::string &model_path) const;
    void load_model(const std::string &model_path);



    // Encoding
    std::vector<int> encode_as_ids(const std::string &text) const;
    std::vector<std::string> encode_as_tokens(const std::string &text) const;

    // Decoding
    std::string decode_ids(const std::vector<int> &ids) const;
    std::string decode_tokens(const std::vector<std::string> &tokens) const;

    // Utility
    int token_to_id(const std::string &piece) const;
    std::string id_to_token(int id) const;
    int get_token_size() const;
    std::vector<std::string> get_vocab() const;

private:
    mutable std::mutex mu;

    // Vocabulary
    std::unordered_map<std::string,int> token2id;
    std::vector<std::string> id2token;

    TokenizerMode mode = TokenizerMode::WORD;

    int unkId;
    int padId;
    int sosId;
    int eosId;

    std::unordered_map<std::string, std::pair<std::string,std::string>> mergeRules;

    void ensure_special_tokens();
};

