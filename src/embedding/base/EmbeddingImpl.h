//
// Created by vlad on 10.04.19.
//

#ifndef EMBEDDINGS_EMBEDDINGIMPL_H
#define EMBEDDINGS_EMBEDDINGIMPL_H

#include "Embedding.h"
#include <unordered_map>
#include <core/vec.h>
#include <vector>
#include <iostream>

template <class T>
class EmbeddingImpl
        : public Embedding<T>{
private:
    std::unordered_map<T, Vec> mapping;
    std::vector<T> vocab;
    std::unordered_map<T, int> invVocab;

public:
    explicit EmbeddingImpl(std::unordered_map<T, Vec> &map);
    bool inVocab(T obj);
    int vocabSize();
    int getIndex(T obj);
    T getObj(int i);
    double distance(T a, T b);

    Vec operator()(T t) override;

    void write (std::ostream &to);

    static EmbeddingImpl& read(std::istream &from);

};


#endif //EMBEDDINGS_EMBEDDINGIMPL_H
