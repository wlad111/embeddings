//
// Created by vlad on 15.04.19.
//

#ifndef EMBEDDINGS_COOCBASEDBUILDER_H
#define EMBEDDINGS_COOCBASEDBUILDER_H

#include "EmbeddingBuilderBase.h"
#include <vector>

//TODO maybe add some typedefs (for example LongSeq for std::vector<int64_t>)
//TODO maybe add logger


class CoocBasedBuilder:EmbeddingBuilderBase {
private:
    const int32_t capacity = 50000000;

    std::vector<std::vector<int64_t>> cooc_;

    bool coocReady = false;

    void acquireCoocurrences();

    void merge(std::vector<int64_t> acc);

protected:
    Embedding<std::string> fit() override ;

    std::vector<std::string> dict();

    int32_t index(std::string word);

    std::vector<int64_t> cooc(size_t i);

    //TODO add here protected synchronized void cooc(int i, LongSeq set)

    float unpackWeight(std::vector<int64_t > &cooc, int32_t v);

    int32_t unpackB(std::vector<int64_t > &cooc, int32_t v);

public:
    Embedding<std::string> build() override;

};


#endif //EMBEDDINGS_COOCBASEDBUILDER_H
