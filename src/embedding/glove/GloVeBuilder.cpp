//
// Created by vlad on 07.04.19.
//

#include "GloVeBuilder.h"
#include <cmath>
#include <algorithm>
#include <execution>
#include <torch/torch.h>

void GloVeBuilder::alpha(double alpha) {
    alpha_ = alpha;
}

void GloVeBuilder::x_max(double x_max) {
    x_max_ = x_max;
}

void GloVeBuilder::dim(int dim) {
    dim_ = dim;
}

std::unique_ptr<Embedding<std::string>> GloVeBuilder::fit() {
    auto start_training = std::chrono::steady_clock::now();
    int vocab_size = dict().size();
    torch::Tensor leftVectors = torch::rand({dim_, vocab_size});
    leftVectors -= 0.5;
    leftVectors /= dim_;

    torch::Tensor rightVectors = torch::rand({dim_, vocab_size});
    rightVectors -= 0.5;
    rightVectors /= dim_;


    torch::Tensor biasLeft = torch::rand({vocab_size});
    biasLeft -= 0.5;
    biasLeft /= dim_;

    torch::Tensor biasRight = torch::rand({vocab_size});
    biasRight -= 0.5;
    biasRight /= dim_;

    torch::Tensor softmaxLeft = torch::ones(leftVectors.sizes());
    torch::Tensor softmaxRight = torch::ones(rightVectors.sizes());

    torch::Tensor softBiasLeft = torch::ones(biasLeft.sizes());
    torch::Tensor softBiasRight = torch::ones(biasRight.sizes());

    std::vector<size_t> vocab_size_range;

    for (size_t i = 0; i < vocab_size; i++) {
        vocab_size_range.push_back(i);
    }

    for (int iter = 0; iter < T(); iter++) {
        auto start = std::chrono::steady_clock::now();

        ScoreCalculator score_calculator(vocab_size);

        //Accessors for efficient iterations through tensors
        auto leftVectors_a = leftVectors.accessor<float, 2>();
        auto rightVectors_a = rightVectors.accessor<float, 2>();
        auto softMaxLeft_a = softmaxLeft.accessor<float, 2>();
        auto softMaxRight_a = softmaxRight.accessor<float, 2>();
        auto biasLeft_a = biasLeft.accessor<float, 1>();
        auto biasRight_a = biasRight.accessor<float, 1>();
        auto softBiasLeft_a = softBiasLeft.accessor<float, 1>();
        auto softBiasRight_a = softBiasRight.accessor<float, 1>();


        std::for_each(std::execution::par_unseq, vocab_size_range.begin(), vocab_size_range.end(), [&](size_t i) mutable {
            const std::vector<CoocBasedBuilder::cooc_token> &coocI = cooc(i);
            std::for_each(std::execution::par_unseq, coocI.begin(), coocI.end(),
                    [&](CoocBasedBuilder::cooc_token packed) mutable {
                int j = packed.j;
                //int j = packed >> 32;
                float X_ij = packed.cooccurence;
                //auto int_to_float = static_cast<int32_t >(packed& 0xFFFFFFFFL);
                //float X_ij = *reinterpret_cast<float*>(&int_to_float);

                double asum = 0;
                for (int k = 0; k < dim_; k++) {
                   asum += leftVectors_a[k][i]*rightVectors_a[k][j];
                }
                double diff = biasLeft_a[i] + biasRight_a[j] + asum - log(X_ij);
                double weight = weightingFunc(X_ij);
                double fdiff = step() * diff * weight;
                score_calculator.adjust(i, j, weight, 0.5 * weight * diff * diff);
                for (int id = 0; id < dim_; id++) {
                    double dL = fdiff * rightVectors_a[id][j]/*right.get(id)*/;
                    double dR = fdiff * leftVectors_a[id][i]/*left.get(id)*/;
                    leftVectors_a[id][i] -= dL / sqrt(softMaxLeft_a[id][i])/*left.set(id, left.get(id) - dL / sqrt(softMaxL.get(id)))*/;
                    rightVectors_a[id][j] -= dR / sqrt(softMaxRight_a[id][j])/*right.set(id, right.get(id) - dR / sqrt(softMaxR.get(id)))*/;
                    softMaxLeft_a[id][i] += dL * dL/*softMaxL.set(id, softMaxL.get(id) + dL * dL)*/;
                    softMaxRight_a[id][j] += dR * dR/*softMaxR.set(id, softMaxR.get(id) + dR * dR)*/;
                }
                biasLeft_a[i] -= fdiff / sqrt(softBiasLeft_a[i]);
                biasRight_a[j] -= fdiff / sqrt(softBiasRight_a[j]);
                softBiasLeft_a[i] += fdiff * fdiff;
                softBiasRight_a[j] += fdiff * fdiff;
            });
        });
        auto end = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Iteration " << iter <<
                  ", score " << score_calculator.gloveScore() <<
                  ", time " << elapsed << " ms " <<
                  std::endl;
    }

    auto end_training = std::chrono::steady_clock::now();
    int elapsed_tr = std::chrono::duration_cast<std::chrono::seconds>(end_training - start_training).count();
    //std::cout << "Trained vectors for " << elapsed_tr << " sec " << std::endl;
    //std::unordered_map<std::string, torch::Tensor> mapping;

    auto leftVectors_t = leftVectors.t();
    auto rightVectors_t = rightVectors.t();

    for (int i = 0; i < dict().size(); i++) {
        std::string word = dict()[i];

        mapping.emplace(word, leftVectors_t[i] + rightVectors_t[i]);
    }
    std::unique_ptr<Embedding<std::string>> result(new EmbeddingImpl(mapping));
    return result;
}

double GloVeBuilder::weightingFunc(double x) {
    return x < x_max_ ? pow(x / x_max_, alpha_) : 1;
}

double GloVeBuilder::initializeValue() {
    return dist(mt);
}

GloVeBuilder::GloVeBuilder(std::string &dictPath) :
    CoocBasedBuilder(dictPath), mt(rd()), dist(-0.5 / dim_, 0.5 / dim_) {
}
