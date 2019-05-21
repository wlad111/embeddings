//
// Created by vlad on 15.04.19.
//

#include "CoocBasedBuilder.h"

void CoocBasedBuilder::acquireCoocurrences() {
    if (!coocReady) {
        std::vector<std::vector<int64_t >> accumulators;

    }
}

void CoocBasedBuilder::merge(std::vector<int64_t> acc) {

}

void CoocBasedBuilder::fit() {
    std::cout << "fit" << std::endl;
    acquireDictionary();
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

void CoocBasedBuilder::build() {

}

CoocBasedBuilder::CoocBasedBuilder(std::string &dict_path) : EmbeddingBuilderBase(dict_path) {
    std::cout << "CoocBasedBuilder" << std::endl;
}

//std::vector<uint64_t> CoocBasedBuilder::positionsStream() {
    std::vector<uint64_t> coocStream;

//}


