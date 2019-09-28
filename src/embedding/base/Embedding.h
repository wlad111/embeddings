//
// Created by vlad on 07.04.19.
//


#ifndef EMBEDDINGS_EMBEDDING_H
#define EMBEDDINGS_EMBEDDING_H

#include <cmath>
#include <torch/torch.h>
#include <memory>


template <class T>
class Embedding {
public:

    virtual torch::Tensor & operator()(T arg) = 0;

    virtual ~Embedding();

    class Builder {
    public:
        virtual std::unique_ptr<Builder> file(const std::string &path) = 0;
        virtual std::unique_ptr<Builder> minWordCount (int count) = 0;
        virtual std::unique_ptr<Builder> iterations (int count) = 0;
        virtual std::unique_ptr<Builder> step (double step) = 0;
        virtual std::unique_ptr<Embedding<T>> build () = 0;
        //virtual void window (WindowType type, int left, int right) = 0;
        virtual ~Builder();
    };

private:
};


#endif //EMBEDDINGS_EMBEDDING_H
