//
// Created by vlad on 10.04.19.
//

#ifndef EMBEDDINGS_EMBEDDINGBUILDERBASE_H
#define EMBEDDINGS_EMBEDDINGBUILDERBASE_H

#include "Embedding.h"
#include <string>
#include <vector>
#include <unordered_map>

class EmbeddingBuilderBase : Embedding<string>::Builder {

private:
    int minCount_ = 5;
    int windowLeft = 15;
    int windowRight = 15;

    typename Embedding<string>::WindowType window_type = Embedding<string>::WindowType::LINEAR;

    int iterations_ = 25;
    double step_ = 0.01;
    bool dictReady;
protected:
    std::vector<string> wordsList;
    //TODO initialize wordsIndex
    std::unordered_map<string, int> wordsIndex;

    virtual Embedding<string> fit() = 0;

    std::vector<string> dict();
    int index(string word);
    int T();
    double step();
    int minCount();
    Embedding<string>::WindowType wtype();
    int wleft();
    int wright();

public:
    void window (Embedding<string>::WindowType type, int left, int right) override;
    void step (double step) override;
    void iterations (int count) override;
    void minWordCount (int count) override;
    Embedding<string> build() override;
    //TODO add other functions & implement them
};


#endif //EMBEDDINGS_EMBEDDINGBUILDERBASE_H