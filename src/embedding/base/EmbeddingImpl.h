//
// Created by vlad on 10.04.19.
//

#ifndef EMBEDDINGS_EMBEDDINGIMPL_H
#define EMBEDDINGS_EMBEDDINGIMPL_H

#include "Embedding.h"
#include <unordered_map>
#include <vector>
#include <iostream>
#include <torch/torch.h>


class EmbeddingImpl
        : public Embedding<std::string>{

public:
    explicit EmbeddingImpl(std::unordered_map<std::string, torch::Tensor> & map);
    bool inVocab(std::string obj);
    int vocabSize();
    int getIndex(std::string obj);
    std::string getObj(int i);
    double distance(std::string a, std::string b);

    //Vec operator()(T t) override;

    void write (std::ostream &to);

    static EmbeddingImpl& read(std::istream &from);

    torch::Tensor & operator()(std::string arg);

private:
    std::unordered_map<std::string, torch::Tensor> mapping;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> invVocab;



};


#endif //EMBEDDINGS_EMBEDDINGIMPL_H
