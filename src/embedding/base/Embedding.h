//
// Created by vlad on 07.04.19.
//

#include <functional>
#include <core/vec.h>
#include <bits/std_function.h>
#include <cmath>

#ifndef EMBEDDINGS_EMBEDDING_H
#define EMBEDDINGS_EMBEDDING_H

template <class T>
class Embedding:
        public std::function<Vec(T)>{
public:

    virtual Vec operator()(T arg) const = 0;

    class Builder {
        virtual void minWordCount (int count) = 0;
        virtual void iterations (int count) = 0;
        virtual void step (double step) = 0;

        virtual Embedding<T> build () = 0;

    };



private:
};


#endif //EMBEDDINGS_EMBEDDING_H
