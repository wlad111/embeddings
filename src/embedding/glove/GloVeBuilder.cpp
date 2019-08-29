//
// Created by vlad on 07.04.19.
//

#include "GloVeBuilder.h"
#include <cmath>
#include <core/matrix.h>
#include <vec_tools/fill.h>
#include <vec_tools/distance.h>

void GloVeBuilder::alpha(double alpha) {
    alpha_ = alpha;
}

void GloVeBuilder::x_max(double x_max) {
    x_max_ = x_max;
}

void GloVeBuilder::dim(int dim) {
    dim_ = dim;
}

void GloVeBuilder::fit() {
    CoocBasedBuilder::fit();
    int vocab_size = dict().size();
    Mx leftVectors(vocab_size, dim_);
    Mx rightVectors(vocab_size, dim_);
    Vec biasLeft(vocab_size);
    Vec biasRight(vocab_size);

    for (int i = 0; i < vocab_size; i++) {
        biasLeft.set(i, initializeValue());
        biasRight.set(i, initializeValue());
        for (int j = 0; j < dim_; j++) {
            leftVectors.set(i, j, initializeValue());
            rightVectors.set(i, j, initializeValue());
        }
    }

    Mx softMaxLeft(leftVectors.ydim(), leftVectors.xdim()); //TODO check rows or columns need to be here
    Mx softMaxRight(rightVectors.ydim(), rightVectors.xdim());
    Vec softBiasLeft(biasLeft.dim());
    Vec softBiasRight(biasRight.dim());
    VecTools::fill(1.0, softMaxLeft);
    VecTools::fill(1.0, softMaxRight);
    VecTools::fill(1.0, softBiasLeft);
    VecTools::fill(1.0, softBiasRight);

    std::vector<size_t> vocab_size_range;

    for (size_t i = 0; i < vocab_size; i++) {
        vocab_size_range.push_back(i);
    }



    for (int iter = 0; iter < T(); iter++) {
        std::for_each(vocab_size_range.begin(), vocab_size_range.end(), [&](size_t i) mutable {
            Vec left(leftVectors.row(i)); //TODO check if it is correct vec by reference
            Vec softMaxL(softMaxLeft.row(i));
            std::for_each(cooc(i).begin(), cooc(i).end(), [&](int64_t packed) mutable {
                int j = packed >> 32;
                auto int_to_float = static_cast<int32_t >(packed& 0xFFFFFFFFL);
                float X_ij = *reinterpret_cast<float*>(&int_to_float);
                Vec right(rightVectors.row(j));
                Vec softMaxR(softMaxRight.row(j));
                double asum = VecTools::dotProduct(left, right);
                double diff = biasLeft.get(i) + biasRight.get(j) + asum - log(X_ij);
                double weight = weightingFunc(X_ij);
                double fdiff = step() * diff * weight;
                for (int id = 0; id < dim_; id++) {
                    double dL = fdiff * right.get(id);
                    double dR = fdiff * left.get(id);
                    left.set(id, left.get(id) - dL / sqrt(softMaxL.get(id)));
                    right.set(id, right.get(id) - dR / sqrt(softMaxR.get(id)));
                    softMaxL.set(id, softMaxL.get(id) + dL * dL);
                    softMaxR.set(id, softMaxR.get(id) + dR * dR);
                }
                biasLeft.set(i, biasLeft.get(i) - fdiff / sqrt(softBiasLeft.get(i)));
                biasRight.set(j, biasRight.get(j) - fdiff / sqrt(softBiasRight.get(j)));
                softBiasLeft.set(i, softBiasLeft.get(i) + fdiff * fdiff);
                softBiasRight.set(j, biasLeft.get(j) + fdiff * fdiff);
            });
        });
        std::cout << "Iteration " << iter << std::endl;
    }

    std::unordered_map<std::string, Vec> mapping;
    for (int i = 0; i < dict().size(); i++) {
        std::string word = dict()[i];

        mapping[word] += leftVectors.row(i) + rightVectors.row(i);
    }
}

double GloVeBuilder::weightingFunc(double x) {
    return x < x_max_ ? pow(x / x_max_, alpha_) : 1;
}

double GloVeBuilder::initializeValue() {
    return dist(mt);
}

GloVeBuilder::GloVeBuilder(string &dictPath) :
    CoocBasedBuilder(dictPath), mt(rd()), dist(-0.5 / dim_, 0.5 / dim_) {
}
