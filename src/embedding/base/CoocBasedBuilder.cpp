//
// Created by vlad on 15.04.19.
//

#include "CoocBasedBuilder.h"

void CoocBasedBuilder::acquireCoocurrences() {

}

void CoocBasedBuilder::merge(std::vector<int64_t> acc) {

}

void CoocBasedBuilder::fit() {

}

std::vector<std::string> CoocBasedBuilder::dict() {
    return EmbeddingBuilderBase::dict();
}

int32_t CoocBasedBuilder::index(std::string word) {
    return EmbeddingBuilderBase::index(word);
}

std::vector<int64_t> CoocBasedBuilder::cooc(size_t i) {
    return std::vector<int64_t>();
}

float CoocBasedBuilder::unpackWeight(std::vector<int64_t> &cooc, int32_t v) {
    return 0;
}

int32_t CoocBasedBuilder::unpackB(std::vector<int64_t> &cooc, int32_t v) {
    return 0;
}

Embedding<string>* CoocBasedBuilder::build() {

}
