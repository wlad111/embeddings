//
// Created by vlad on 10.04.19.
//

#ifndef EMBEDDINGS_EMBEDDINGBUILDERBASE_H
#define EMBEDDINGS_EMBEDDINGBUILDERBASE_H

#include "Embedding.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>



//TODO add path field, use std::filesystem::path
//TODO add destructor
class EmbeddingBuilderBase
        : public Embedding<string>::Builder {

private:
    int minCount_ = 5;



    int iterations_ = 25;
    double step_ = 0.01;
    bool dictReady = false;
protected:
    std::vector<string> wordsList;
    //TODO initialize wordsIndex
    std::unordered_map<string, int> wordsIndex;

    std::string path_;

    virtual void fit() = 0;

    std::vector<string> dict();
    int index(string word);
    int T();
    double step();
    int minCount();
    Embedding<string>::WindowType wtype();
    int wleft();
    int wright();


    //TODO files processing
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

    //TODO check all packing and unpacking functions
    int32_t unpackA(int64_t next);
    int32_t unpackB(int64_t next);

    double unpackWeight(int64_t next);

    int32_t unpackDist(int64_t next);


    int windowRight = 15;
    int windowLeft = 15;


public:
    EmbeddingBuilderBase(std::string &s);
    void window (Embedding<string>::WindowType type, int left, int right) override;
    void step (double step) override;
    void iterations (int count) override;
    void minWordCount (int count) override;
    void build() override;
    void file(const std::string &path) override;
    //TODO add other functions & implement them

};


#endif //EMBEDDINGS_EMBEDDINGBUILDERBASE_H
