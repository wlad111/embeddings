//
// Created by vlad on 10.04.19.
//

#include "EmbeddingBuilderBase.h"
#include <cctype>
#include <algorithm>

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
//TODO implement acquireDictionary()
}

std::string EmbeddingBuilderBase::normalize(std::string word) {
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

Embedding<string> EmbeddingBuilderBase::build() {
    //TODO this is what I have soon to do
}

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
