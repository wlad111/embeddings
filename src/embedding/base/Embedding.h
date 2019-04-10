//
// Created by vlad on 07.04.19.
//


#ifndef EMBEDDINGS_EMBEDDING_H
#define EMBEDDINGS_EMBEDDING_H

#include <functional>
#include <core/vec.h>
#include <bits/std_function.h>
#include <cmath>


template <class T>
class Embedding:
        public std::function<Vec(T)>{
public:

    virtual Vec operator()(T arg) const = 0;

    enum class Type {
        GLOVE,
        DECOMP,
        MULTI_DECOMP,
        KMEANS_SKIP
    };

    enum class WindowType {
        LINEAR,
        FIXED,
        EXP
    };

    class Builder {
        //TODO add function file()
        virtual void minWordCount (int count) = 0;
        virtual void iterations (int count) = 0;
        virtual void step (double step) = 0;
        virtual Embedding<T> build () = 0;
        virtual void window (WindowType type, int left, int right) = 0;
    };








private:
};


#endif //EMBEDDINGS_EMBEDDING_H
