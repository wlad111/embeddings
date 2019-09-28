//
// Created by vlad on 10.04.19.
//

#ifndef EMBEDDINGS_EMBEDDINGBUILDERBASE_H
#define EMBEDDINGS_EMBEDDINGBUILDERBASE_H

#include "Embedding.h"
#include "EmbeddingImpl.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <torch/torch.h>
//#include <core/vec.h>



//TODO add path field, use std::filesystem::path
class EmbeddingBuilderBase
: public Embedding<std::string>::Builder {

public:
    EmbeddingBuilderBase(std::string &s);
    //void window (Embedding<std::string>::WindowType type, int left, int right) override;
    std::unique_ptr<Builder> step (double step) override;
    std::unique_ptr<Builder> iterations (int count) override;
    std::unique_ptr<Builder> minWordCount (int count) override;
    std::unique_ptr<Embedding<std::string>> build() override;
    std::unique_ptr<Builder> file(const std::string &path) override;
    //TODO add other functions & implement them
    //const std::unordered_map<string, Vec> &get_mapping() const;
    void write_mapping(std::string path);
    std::vector<std::string> closest_words_except(std::string word, int top, std::vector<std::string> except_words);


protected:
    std::vector<std::string> wordsList;
    std::unordered_map<std::string, int> wordsIndex;
    std::unordered_map<std::string, torch::Tensor> mapping;
    std::string path_;

    virtual std::unique_ptr<Embedding<std::string>> fit() = 0;

    std::vector<std::string> dict();
    int index(std::string word);
    int T();
    double step();
    int minCount();
    //Embedding<std::string>::WindowType wtype();
    int wleft();
    int wright();


    void acquireDictionary();
    std::string normalize(std::string &word);
    bool anyLetter(std::string &word);

    std::vector<int64_t> positionsStream();

    class ScoreCalculator {
    private:
        std::vector<double> scores;
        std::vector<double> weights;
        std::vector<int64_t> counts;
    public:
        explicit ScoreCalculator(int dim);
        void adjust(int i, int j, double weight, double value);
        double gloveScore();
        int64_t  count();
    };

    int64_t pack(int64_t a, int64_t b, int8_t dist);

    int32_t unpackA(int64_t next);
    int32_t unpackB(int64_t next);

    double unpackWeight(int64_t next);

    int32_t unpackDist(int64_t next);


    int windowRight = 15;
    int windowLeft = 15;

private:
    int minCount_ = 1;
    int iterations_ = 25;
    double step_ = 0.01;
    bool dictReady = false;

};


#endif //EMBEDDINGS_EMBEDDINGBUILDERBASE_H
