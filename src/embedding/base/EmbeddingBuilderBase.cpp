//
// Created by vlad on 10.04.19.
//

#include "EmbeddingBuilderBase.h"
#include <cctype>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>


std::vector<string> EmbeddingBuilderBase::dict() {
    return wordsList;
}

int EmbeddingBuilderBase::index(string word) {
    return wordsIndex[word];
}

int EmbeddingBuilderBase::T() {
    return iterations_;
}

double EmbeddingBuilderBase::step() {
    return step_;
}

int EmbeddingBuilderBase::minCount() {
    return minCount_;
}



int EmbeddingBuilderBase::wleft() {
    return windowLeft;
}

int EmbeddingBuilderBase::wright() {
    return windowRight;
}

void EmbeddingBuilderBase::acquireDictionary() {
    std::cout << "acquiring dictionary" << std::endl;
//TODO add reading from file
    if (!dictReady) {
        std::cout << "====Generating dictionary====" << std::endl;

        auto start = std::chrono::steady_clock::now();


        std::ifstream source(path_);
        std::unordered_map<std::string, int32_t> wordsCount;
        std::string word;
        int64_t i = 0;
        for (source >> word; !source.eof(); source >> word) {
            if (i % 100000 == 0) {
                std::cout << i << " words processed" << std::endl;
            }
            if (anyLetter(word)) {
                word = normalize(word);
                wordsCount[word]++;
            }
            i++;
        }
        std::cout << i << " words processed" << std::endl;
        std::vector<std::pair<int32_t, std::string>> words;

        for (auto entry : wordsCount) {
            words.emplace_back(std::make_pair(entry.second, entry.first));
        }

        std::sort(words.begin(), words.end(), std::greater<>());

        auto end = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Generated dictionary for " << elapsed << "ms" << std::endl;


        std::cout << "====Writing dictionary to file====" << std::endl;
        std::string dict_path = "dict.csv";
        std::ofstream dictfs(dict_path);
        dictfs << "word" << "," << "freq" << std::endl;
        for (auto word:words) {
            dictfs << word.second << "," << word.first << std::endl;
        }

        for (auto word : words) {
            if (wordsCount[word.second] >= minCount_) {
                wordsIndex[word.second] = wordsList.size();
                wordsList.push_back(word.second);
            }
        }
        dictReady = true;




    }
}

bool EmbeddingBuilderBase::anyLetter(std::string &word) {
    for (auto c: word) {
        if (isalpha(c)) {
            return true;
        }
    }
    return false;
}

std::string EmbeddingBuilderBase::normalize(std::string &word) {
    size_t initialLength = word.size();
    size_t len = initialLength;
    int32_t st = 0;
    while ((st < len) && (!isalnum(word[st]))){
        st++;
    }
    while ((st < len) && (!isalnum(word[len - 1]))){
        len--;
    }

    word = ((st > 0) || (len < initialLength)) ? word.substr(st, len - st) : word;
    std::transform(word.begin(), word.end(), word.begin(), ::tolower);
    return word;
}

void EmbeddingBuilderBase::window(Embedding<string>::WindowType type, int left, int right) {
    windowLeft = left;
    windowRight = right;
}

void EmbeddingBuilderBase::step(double step){
    step_ = step;
}

void EmbeddingBuilderBase::iterations(int count) {
    iterations_ = count;
}

void EmbeddingBuilderBase::minWordCount(int count) {
    minCount_ = count;
}

void EmbeddingBuilderBase::build() {
    std::cout << "==== Dictionary phase ====" << std::endl;
    auto start = std::chrono::system_clock::now();
    acquireDictionary();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "==== " << elapsed_seconds.count() << "s ====" << std::endl;
}

void EmbeddingBuilderBase::file(const std::string &path) {
    path_ = path;
}

int64_t EmbeddingBuilderBase::pack(int64_t a, int64_t b, int8_t dist) {
    return (a << 36) | (b << 8) | (static_cast<uint64_t> (dist & 0xFF));
}

int32_t EmbeddingBuilderBase::unpackA(int64_t next) {
    return (int)(next >> 36);
}

int32_t EmbeddingBuilderBase::unpackB(int64_t next) {
    return ((int)(next >> 8)) & 0x0FFFFFFF;
}

//double EmbeddingBuilderBase::unpackWeight(int64_t next) {
//    int dist = (int) (0xFF & next);
//    return wtype().weight(dist > 126 ? -256 + dist : dist);
//}

int32_t EmbeddingBuilderBase::unpackDist(int64_t next) {
    return (int)(0xFF & next);
}

std::vector<int64_t> EmbeddingBuilderBase::positionsStream() {
    std::cout << "positions stream" << std::endl;
    std::ifstream source(path_);
    //std::string line;
    //std::string newLine = "777newline777";
    int64_t nLine = 0;

    std::vector<int32_t> pos_queue;
    int32_t offset = 0;

    std::vector<int64_t> res;
    std::string word;
    int64_t i = 0;
    for (source >> word; !source.eof(); source >> word) {
        if (i % 100000 == 0) {
            std::cout << i << " words processed" << std::endl;
        }
        int32_t idx = wordsIndex[normalize(word)];

        int32_t pos = pos_queue.size();
        int64_t out[windowRight + windowLeft];
        int32_t outIndex = 0;
        for (int j = offset; j < pos; j++) {
            int8_t distance = pos - j;
            if (distance == 0) {
                std::cout << "Zero distance occured! pos: " << pos << "i: " << j << std::endl;
            }
            if (distance <= windowRight) {
                out[outIndex++] = pack(pos_queue[j], idx, distance);
            }
            if (distance <= windowLeft) {
                out[outIndex++] = pack(idx, pos_queue[j], -distance);
            }
        }
        pos_queue.push_back(idx);
        if (pos_queue.size() > std::max(windowLeft, windowRight)) {
            offset++;
            if (offset > 1000 - std::max(windowLeft, windowRight)) {
                pos_queue.erase(pos_queue.begin(), pos_queue.begin() + offset);
                offset = 0;
            }
        }
        for (auto entry: out) {
            res.push_back(entry);
        }
        i++;
    }


//    while (std::getline(source, line)){
//        if ((++nLine) % 10000 == 0) {
//            std::cout << nLine << " lines processed" << std::endl;
//        }
//        line += newLine;
//
//    }
    return res;
}

EmbeddingBuilderBase::EmbeddingBuilderBase(std::string &s) {
    std::cout << "EmbeddingBuilderBase" << std::endl;
    path_ = s;
}

double EmbeddingBuilderBase::unpackWeight(int64_t next) {
    return 0;
}



//std::ifstream EmbeddingBuilderBase::source(std::string path) {
//
//}

EmbeddingBuilderBase::ScoreCalculator::ScoreCalculator(int dim){
    counts.resize(dim);
    scores.resize(dim);
    weights.resize(dim);
}

void EmbeddingBuilderBase::ScoreCalculator::adjust(int i, int j, double weight, double value) {
    weights[i] += weight;
    scores[i] += value;
    counts[i] ++;
}

double EmbeddingBuilderBase::ScoreCalculator::gloveScore() {
    double sum_scores = 0;
    int64_t sum_counts = 0;
    for (auto score : scores) {
        sum_scores += score;
    }
    for (auto count : counts) {
        sum_counts += count;
    }
    return sum_scores / sum_counts;
}

int64_t EmbeddingBuilderBase::ScoreCalculator::count() {
    int64_t  sum_counts = 0;
    for (auto count : counts) {
        sum_counts += count;
    }
    return sum_counts;
}
