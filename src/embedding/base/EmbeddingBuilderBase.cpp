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

Embedding<string>::WindowType EmbeddingBuilderBase::wtype() {
    return window_type;
}

int EmbeddingBuilderBase::wleft() {
    return windowLeft;
}

int EmbeddingBuilderBase::wright() {
    return windowRight;
}

void EmbeddingBuilderBase::acquireDictionary() {
//TODO add reading from file
    if (!dictReady) {
        std::cout << "Generating dictionary" << std::endl;
        std::ifstream source(path_);
        std::unordered_map<std::string, int32_t> wordsCount;
        std::string word;
        for (source >> word; !source.eof(); source >> word) {
            word = normalize(word);
            wordsCount[word]++;
            //TODO add filter(letter)
        }

        std::vector<std::pair<int32_t, std::string>> words;

        for (auto entry : wordsCount) {
            words.push_back(std::make_pair(entry.second, entry.first));
        }

        std::sort(words.begin(), words.end(), std::greater<>());

        //TODO write this to file

        for (auto word : words) {
            if (wordsCount[word.second] >= minCount_) {
                wordsIndex[word.second] = wordsList.size();
                wordsList.push_back(word.second);
            }
        }
        dictReady = true;

    }
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

    word = ((st > 0) || (len < initialLength)) ? word.substr(st, len - st + 1) : word;
    std::transform(word.begin(), word.end(), word.begin(), ::tolower);
    return word;
}

void EmbeddingBuilderBase::window(Embedding<string>::WindowType type, int left, int right) {
    windowLeft = left;
    windowRight = right;
    window_type = type;
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

Embedding<string>* EmbeddingBuilderBase::build() {
    std::cout << "==== Dictionary phase ====" << std::endl;
    auto start = std::chrono::system_clock::now();
    acquireDictionary();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "==== " << elapsed_seconds.count() << "s ====" << std::endl;
    return &this;
}

void EmbeddingBuilderBase::file(const std::string &path) {
    path_ = path;
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
