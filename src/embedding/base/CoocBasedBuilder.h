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
private:
    const int32_t capacity = 5'000'000;
    const int32_t accs_count = 10;

    int32_t dense_count_ = 1000;

    //TODO reorganize coocurrence matrix;
    std::vector<std::vector<int64_t>> cooc_;

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

protected:

    std::vector<std::string> dict();

    int32_t index(std::string word);

    const std::vector<int64_t> &cooc(size_t i) const;

    virtual std::unique_ptr<Embedding<std::string>> fit() override;

    float unpackWeight(std::vector<int64_t > &cooc, int32_t v);


};


#endif //EMBEDDINGS_COOCBASEDBUILDER_H
