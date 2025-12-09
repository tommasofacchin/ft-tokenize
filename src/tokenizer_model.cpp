//src/tokenizer_model.cpp
#include "tokenizer_model.hpp"
#include <fstream>   
#include <sstream>     
#include <algorithm> 
#include <mutex>
#include <map>

using namespace std;

TokenizerModel::TokenizerModel()
    : unkId(-1), padId(-1), sosId(-1), eosId(-1)
{
    id2token.reserve(1024);   
    token2id.reserve(1024);
    ensure_special_tokens();
}

void TokenizerModel::ensure_special_tokens() {

    if (!id2token.empty()) return;

    const vector<string> specials = {"<pad>", "<unk>", "<sos>", "<eos>"};

    for (const string &tok : specials) {
        int newId = id2token.size();
        id2token.push_back(tok);
        token2id.emplace(tok, newId);
    }

    padId = token2id.at("<pad>");
    unkId = token2id.at("<unk>");
    sosId = token2id.at("<sos>");
    eosId = token2id.at("<eos>");
}


// Utility
int TokenizerModel::token_to_id(const string &token) const {
    lock_guard<recursive_mutex> lock(mu);

    auto i = token2id.find(token);

    if (i != token2id.end())
        return i->second;

    return unkId;
}

string TokenizerModel::id_to_token(int id) const{
    lock_guard<recursive_mutex> lock(mu);

    if (id >= 0 && id < id2token.size())
        return id2token[id];

    return id2token[unkId];
}

int TokenizerModel::get_token_size() const{
    lock_guard<recursive_mutex> lock(mu);
    return id2token.size();
}

vector<string> TokenizerModel::get_vocab() const{
    lock_guard<recursive_mutex> lock(mu);
    return id2token;
}


// Encoding
vector<int> TokenizerModel::encode_as_ids(const string &text) const {
    lock_guard<recursive_mutex> lock(mu);

    vector<int> ids;

    if (mode == TokenizerMode::WORD) {
        istringstream stream(text);
        string token;
        while (stream >> token) {
            auto it = token2id.find(token);
            ids.push_back(it != token2id.end() ? it->second : unkId);
        }
    } else if (mode == TokenizerMode::BPE) {
        size_t start = 0;
        while (start < text.size()) {
            size_t end = text.size();
            int id = -1;
            string bestMatch;

            while (end > start) {
                string piece = text.substr(start, end - start);
                auto it = token2id.find(piece);
                if (it != token2id.end()) {
                    bestMatch = piece;
                    id = it->second;
                    break;
                }
                end--;
            }

            if (id == -1) {
                ids.push_back(unkId);
                start++;
            } else {
                ids.push_back(id);
                start += bestMatch.size();
            }
        }
    }

    return ids;
}


vector<string> TokenizerModel::encode_as_tokens(const string &text) const {
    lock_guard<recursive_mutex> lock(mu);

    vector<string> tokens;

    if (mode == TokenizerMode::WORD) {
        istringstream stream(text);
        string token;
        while (stream >> token) {
            if (token2id.find(token) != token2id.end())
                tokens.push_back(token);
            else
                tokens.push_back("<unk>");
        }
    } else if (mode == TokenizerMode::BPE) {
        size_t start = 0;
        while (start < text.size()) {
            size_t end = text.size();
            string bestMatch;
            
            while (end > start) {
                string piece = text.substr(start, end - start);
                if (token2id.find(piece) != token2id.end()) {
                    bestMatch = piece;
                    break;
                }
                end--;
            }

            if (bestMatch.empty()) {
                tokens.push_back("<unk>");
                start++;
            } else {
                tokens.push_back(bestMatch);
                start += bestMatch.size();
            }
        }
    }

    return tokens;
}


// Decoding
string TokenizerModel::decode_ids(const vector<int> &ids) const {
    lock_guard<recursive_mutex> lock(mu);

    string text;

    for (size_t i = 0; i < ids.size(); ++i) {
        int id = ids[i];
        string token;

        if (id >= 0 && id < id2token.size())
            token = id2token[id];
        else
            token = "<unk>";

        if (mode == TokenizerMode::WORD) {
            if (!text.empty()) text += " ";
            text += token;
        } else if (mode == TokenizerMode::BPE) {
            text += token; 
        }
    }

    return text;
}


string TokenizerModel::decode_tokens(const vector<string> &tokens) const {
    lock_guard<recursive_mutex> lock(mu);

    string text;

    for (const auto &token : tokens) {
        string t = token2id.find(token) != token2id.end() ? token : "<unk>";

        if (mode == TokenizerMode::WORD) {
            if (!text.empty()) text += " ";
            text += t;
        } else if (mode == TokenizerMode::BPE) {
            text += t;
        }
    }

    return text;
}


 

// Training
void TokenizerModel::train_from_textfile(const string &input_file,
                             size_t vocab_size,
                             const vector<string> &user_defined_symbols,
                             TokenizerMode mode){
    
    this->mode = mode;

    if (mode == TokenizerMode::WORD) {
        train_word_level(input_file, vocab_size, user_defined_symbols);
    } else if (mode == TokenizerMode::BPE) {
        train_bpe(input_file, vocab_size, user_defined_symbols);
    }

    
}

bool compareByFreq(const pair<string, size_t> &a, const pair<string, size_t> &b) {
    return a.second > b.second; 
}

void TokenizerModel::train_word_level(const string &input_file,
                                      size_t vocab_size,
                                      const vector<string> &user_defined_symbols){

    lock_guard<recursive_mutex> lock(mu);

    // Load the file
    ifstream file(input_file);
    if (!file.is_open()) {
        throw runtime_error("Impossible to open file");
    }


    // Frequency map
    map<string, size_t> freqMap;
    string line;

    while (getline(file, line)) {

        istringstream stream(line);
        string token;

        while (stream >> token) {
            freqMap[token]++;
        }

    }


    // Order tokens by frequency
    vector<pair<string, size_t>> freqVec(freqMap.begin(), freqMap.end());
    sort(freqVec.begin(), freqVec.end(), compareByFreq);


    // Reset vocabulary
    id2token.clear();
    token2id.clear();
    ensure_special_tokens();


    // Add most frequent tokens
    size_t toAdd = vocab_size;

    for (const auto &p : freqVec) {
        const string &token = p.first;

        if (token2id.find(token) == token2id.end()){
            id2token.push_back(token);
            token2id[token] = id2token.size() - 1;

            if(--toAdd == 0) break;
        }

    }


    // Add user defined symbols
    for (const auto &token : user_defined_symbols) {
        if (token2id.find(token) == token2id.end()) {
            id2token.push_back(token);
            token2id[token] = id2token.size() - 1;
        }
    }
}

void TokenizerModel::train_bpe(const string &input_file,
                               size_t vocab_size,
                               const vector<string> &user_defined_symbols) {

    lock_guard<recursive_mutex> lock(mu);

    // Load the file
    ifstream file(input_file);
    if (!file.is_open()) throw runtime_error("Impossible to open file");


    // Frequency map
    //unordered_map<vector<string>, size_t> wordFreqVec;
    map<vector<string>, size_t> wordFreqVec;

    string line;
    while (getline(file, line)) {
        istringstream stream(line);
        string token;
        while (stream >> token) {
            vector<string> symbols;
            for (char c : token) symbols.push_back(string(1, c));
            wordFreqVec[symbols]++;
        }
    }
    file.close();

    // Initialize vocabulary with characters
    id2token.clear();
    token2id.clear();
    ensure_special_tokens();

    for (const auto &p : wordFreqVec) {
        for (const string &ch : p.first) {
            if (token2id.find(ch) == token2id.end()) {
                int newId = id2token.size();
                id2token.push_back(ch);
                token2id[ch] = newId;
            }
        }
    }

    struct pair_hash {
        size_t operator()(const pair<string,string>& p) const {
            return hash<string>()(p.first) ^ (hash<string>()(p.second) << 1);
        }
    };

    size_t merges_done = 0;  
    const size_t MAX_MERGES = 50000;

    // Combine frequent pairs
    while (id2token.size() < vocab_size) {
        
        if (merges_done++ > MAX_MERGES) break; //Loop

        unordered_map<pair<string,string>, size_t, pair_hash> pairFreq;

        for (const auto &p : wordFreqVec) {
            const vector<string> &symbols = p.first;
            size_t freq = p.second;

            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                pairFreq[{symbols[i], symbols[i+1]}] += freq;
            }
        }

        if (pairFreq.empty()) break;

        // Fine most frequent pair
        pair<string,string> bestPair;
        size_t maxFreq = 0;
        for (const auto &p : pairFreq) {
            if (p.second > maxFreq) {
                maxFreq = p.second;
                bestPair = p.first;
            }
        }

        string merge = bestPair.first + bestPair.second;

        // Add to vocabulary
        if (token2id.find(merge) == token2id.end()) {
            int newId = id2token.size();
            id2token.push_back(merge);
            token2id[merge] = newId;
        }

        // Update wordFreqVec with merged pair
        map<vector<string>, size_t> newWordFreqVec;
        
        for (const auto &p : wordFreqVec) {
            const vector<string> &symbols = p.first;
            size_t freq = p.second;
            vector<string> newSymbols;

            for (size_t i = 0; i < symbols.size(); ) {
                if (i + 1 < symbols.size() && symbols[i] == bestPair.first && symbols[i+1] == bestPair.second) {
                    newSymbols.push_back(merge);
                    i += 2;
                } else {
                    newSymbols.push_back(symbols[i]);
                    i += 1;
                }
            }

            newWordFreqVec[newSymbols] = freq;
        }

        wordFreqVec = move(newWordFreqVec);
    }

    // Add user defined symbols
    for (const auto &token : user_defined_symbols) {
        if (token2id.find(token) == token2id.end()) {
            int newId = id2token.size();
            id2token.push_back(token);
            token2id[token] = newId;
        }
    }
}




void TokenizerModel::save_model(const string &model_path) const{
    lock_guard<recursive_mutex> lock(mu);

    ofstream ofs(model_path);
    if (!ofs.is_open()) {
        throw runtime_error("Cannot open file");
    }

    for (const auto &token : id2token) {
        ofs << token << "\n";
    }

    ofs.close();
}

void TokenizerModel::load_model(const string &model_path){
    lock_guard<recursive_mutex> lock(mu);

    ifstream ifs(model_path);
    if (!ifs.is_open()) {
        throw runtime_error("Cannot open file");
    }

    id2token.clear();
    token2id.clear();

    string token;
    while (getline(ifs, token)) {
        if (!token.empty()) {
            int newId = id2token.size();
            id2token.push_back(token);
            token2id[token] = newId;
        }
    }

    padId = token2id.count("<pad>") ? token2id.at("<pad>") : -1;
    unkId = token2id.count("<unk>") ? token2id.at("<unk>") : -1;
    sosId = token2id.count("<sos>") ? token2id.at("<sos>") : -1;
    eosId = token2id.count("<eos>") ? token2id.at("<eos>") : -1;

    ifs.close();
}

