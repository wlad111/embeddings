//
// Created by vlad on 07.04.19.
//

#include "Embedding.h"

template<class T>
Embedding<T>::Builder::~Builder() {

}

template<class T>
Embedding<T>::~Embedding() {

}

template<>
Embedding<std::string>::~Embedding() {

}

template<>
Embedding<std::string>::Builder::~Builder() {

}