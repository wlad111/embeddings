//
// Created by vlad on 10.04.19.
//

//TODO define write() and read()

#include "EmbeddingImpl.h"

EmbeddingImpl::EmbeddingImpl(std::unordered_map<std::string, torch::Tensor> & map) {
    mapping = map;

    for(auto kv : map) {
        vocab.push_back(kv.first);
    }

    for (size_t i = 0; i < vocab.size(); i++) {
        invVocab[vocab[i]] = i;
    }
}

torch::Tensor & EmbeddingImpl::operator()(std::string arg) {
    return mapping[arg];
}





/*template<class T>
bool EmbeddingImpl<T>::inVocab(T obj) {
    for (auto entry:vocab) {
        if (entry == obj) {
            return true;
        }
    }
    return false;
}*/

//template<class T>
/*
int EmbeddingImpl<T>::vocabSize() {
    return vocab.size();
}
*/

//template<class T>
/*int EmbeddingImpl<T>::getIndex(T obj) {
    return invVocab[obj];
}*/

/*template<class T>
T EmbeddingImpl<T>::getObj(int i) {
    return vocab[i];
}

template<class T>
double EmbeddingImpl<T>::distance(T a, T b) {
    Vec vA = mapping[a];
    Vec vB = mapping[b];
    Scalar dot_prod(VecTools::dotProduct(vA, vB));
    Scalar normA(VecTools::norm(vA));
    Scalar normB(VecTools::norm(vB));
    return (double(dot_prod) != 0 ? double(dot_prod) / (double(normA) * double(normB)) : 0);
}

template<class T>
Vec EmbeddingImpl<T>::operator()(T t) {
    return mapping[t];
}*/






