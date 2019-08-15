//
// Created by vlad on 15.04.19.
//

#ifndef EMBEDDINGS_COOCBASEDBUILDER_H
#define EMBEDDINGS_COOCBASEDBUILDER_H

#include "EmbeddingBuilderBase.h"
#include <vector>
#include <condition_variable>
#include <mutex>

//TODO maybe add some typedefs (for example LongSeq for std::vector<int64_t>)
//TODO maybe add logger


class CoocBasedBuilder
        : public EmbeddingBuilderBase {
private:
    const int32_t capacity = 50000000;

    int32_t dense_count_ = 1000;

    std::vector<std::vector<int64_t>> cooc_;

    bool coocReady = false;

    void acquireCoocurrences();

    void merge(std::vector<int64_t> acc);

    std::mutex mt;
    std::condition_variable cv;
    int bufferSize = 1'000'000;
    std::deque<std::string> buffer;

    void readWords (std::ifstream& reader);
    void sendWords();
    void processWords(std::deque<std::string>& buf);
    int countwords = 0;

protected:

    std::vector<std::string> dict();

    int32_t index(std::string word);

    std::vector<int64_t> cooc(size_t i);

    //TODO add here protected synchronized void cooc(int i, LongSeq set)

    float unpackWeight(std::vector<int64_t > &cooc, int32_t v);

    int32_t unpackB(std::vector<int64_t > &cooc, int32_t v);


public:
    void build() override;
    void fit() override;
    explicit CoocBasedBuilder(std::string &dict_path);
};


#endif //EMBEDDINGS_COOCBASEDBUILDER_H
