//
// Created by vlad on 15.04.19.
//

#ifndef EMBEDDINGS_COOCBASEDBUILDER_H
#define EMBEDDINGS_COOCBASEDBUILDER_H

#include "EmbeddingBuilderBase.h"
#include <vector>
#include <condition_variable>
#include <mutex>
#include <deque>

//TODO maybe add logger


class CoocBasedBuilder
        : public EmbeddingBuilderBase {

public:
    std::unique_ptr<Embedding<std::string>> build();
    explicit CoocBasedBuilder(std::string &dict_path);

    virtual ~CoocBasedBuilder();


protected:


    struct cooc_token {
        float cooccurence;
        int64_t j;
    };

    std::vector<std::string> dict();

    int32_t index(std::string word);

    const std::vector<cooc_token> &cooc(size_t i) const;

    virtual std::unique_ptr<Embedding<std::string>> fit() override;

    float unpackWeight(std::vector<int64_t > &cooc, int32_t v);

private:


    const int32_t capacity = 5'000'000;
    const int32_t accs_count = 10;

    int32_t dense_count_ = 1000;

    std::vector<std::vector<int64_t>> cooc_;

    std::vector<std::vector<cooc_token>> cooc_1;

    bool coocReady = false;

    void acquireCoocurrences();

    void merge(std::vector<int64_t> &acc);

    //std::mutex mt;
    std::condition_variable cv;
    int bufferSize = 100'000'000;
    std::deque<std::string> buffer;
    std::vector<std::vector<int64_t >> accumulators;


    void readWords (std::ifstream& reader);
    void sendWords();
    void processWords(std::deque<std::string>& buf);
    int countwords = 0;

};


#endif //EMBEDDINGS_COOCBASEDBUILDER_H
