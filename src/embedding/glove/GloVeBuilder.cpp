//
// Created by vlad on 07.04.19.
//

#include "GloVeBuilder.h"
#include <cmath>
#include <algorithm>
#include <execution>
#include <torch/torch.h>s

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
    at::Tensor left_vectors = at::rand({vocab_size, dim_});

    left_vectors -= 0.5;
    left_vectors /= dim_;

    at::Tensor right_vectors = at::rand({vocab_size, dim_});
    right_vectors -= 0.5;
    right_vectors /= dim_;


    at::Tensor bias_left = at::rand({vocab_size});
    bias_left -= 0.5;
    bias_left /= dim_;

    at::Tensor bias_right = at::rand({vocab_size});
    bias_right -= 0.5;
    bias_right /= dim_;

    at::Tensor softmax_left = at::ones(left_vectors.sizes());
    at::Tensor softmax_right = at::ones(right_vectors.sizes());

    at::Tensor soft_bias_left = at::ones(bias_left.sizes());
    at::Tensor soft_bias_right = at::ones(bias_right.sizes());

    left_vectors.to(torch::kCUDA);
    right_vectors.to(torch::kCUDA);
    bias_left.to(torch::kCUDA);
    bias_right.to(torch::kCUDA);
    softmax_left.to(torch::kCUDA);
    softmax_right.to(torch::kCUDA);
    soft_bias_left.to(torch::kCUDA);
    soft_bias_right.to(torch::kCUDA);

    std::vector<size_t> vocab_size_range;



    for (size_t i = 0; i < vocab_size; i++) {
        vocab_size_range.push_back(i);
    }

    for (int iter = 0; iter < T(); iter++) {
        auto start = std::chrono::steady_clock::now();

        ScoreCalculator score_calculator(vocab_size);

        //Accessors for efficient iterations through tensors
        auto left_vectors_a = left_vectors.accessor<float, 2>();
        auto right_vectors_a = right_vectors.accessor<float, 2>();
        auto softmax_left_a = softmax_left.accessor<float, 2>();
        auto softmax_right_a = softmax_right.accessor<float, 2>();
        auto bias_left_a = bias_left.accessor<float, 1>();
        auto bias_right_a = bias_right.accessor<float, 1>();
        auto soft_bias_left_a = soft_bias_left.accessor<float, 1>();
        auto soft_bias_right_a = soft_bias_right.accessor<float, 1>();

        at::Tensor product = at::dot(left_vectors[0], right_vectors[0]);
        product.to(torch::kCUDA);
        //auto product_a = product.data<float>();



        std::for_each(std::execution::par, vocab_size_range.begin(), vocab_size_range.end(), [&](size_t i) mutable {
            const std::vector<int64_t> &coocI = cooc(i);
            std::for_each(std::execution::seq, coocI.begin(), coocI.end(), [&](int64_t packed) mutable {
                int j = packed >> 32;
                auto int_to_float = static_cast<int32_t >(packed& 0xFFFFFFFFL);
                float X_ij = *reinterpret_cast<float*>(&int_to_float);
                at::dot_out(product, left_vectors[i], right_vectors[j]);
                //product = at::dot(left_vectors[i], right_vectors[j]);
                /*std::cout << i << " " << j << std::endl;
                std::cout << product << std::endl;*/

                double asum = 0;  
                /*for (int k = 0; k < dim_; k++) {
                   asum += left_vectors_a[i][k]*right_vectors_a[j][k];
                }*/
                //double diff = bias_left_a[i] + bias_right_a[j] + asum - log(X_ij);
                auto 
                double weight = weightingFunc(X_ij);
                double fdiff = step() * diff * weight;
                score_calculator.adjust(i, j, weight, 0.5 * weight * diff * diff);

                at::Tensor dL = fdiff * right_vectors[j];
                at::Tensor dR = fdiff * left_vectors[i];
                left_vectors[i] -= dL / at::sqrt(softmax_left[i]);
                right_vectors[j] -= dR / at::sqrt(softmax_right[j]);
                softmax_left[i] += dL * dL;
                softmax_right[j] += dR * dR;


                /*for (int id = 0; id < dim_; id++) {
                    double dL = fdiff * right_vectors_a[j][id]*//*right.get(id)*//*;
                    double dR = fdiff * left_vectors_a[i][id]*//*left.get(id)*//*;
                    left_vectors_a[i][id] -= dL / sqrt(softmax_left_a[i][id])*//*left.set(id, left.get(id) - dL / sqrt(softMaxL.get(id)))*//*;
                    right_vectors_a[j][id] -= dR / sqrt(softmax_right_a[j][id])*//*right.set(id, right.get(id) - dR / sqrt(softMaxR.get(id)))*//*;
                    softmax_left_a[i][id] += dL * dL*//*softMaxL.set(id, softMaxL.get(id) + dL * dL)*//*;
                    softmax_right_a[j][id] += dR * dR*//*softMaxR.set(id, softMaxR.get(id) + dR * dR)*//*;
                }*/
                bias_left_a[i] -= fdiff / sqrt(soft_bias_left_a[i]);
                bias_right_a[j] -= fdiff / sqrt(soft_bias_right_a[j]);
                soft_bias_left_a[i] += fdiff * fdiff;
                soft_bias_right_a[j] += fdiff * fdiff;
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
    std::cout << "Trained vectors for " << elapsed_tr << " sec " << std::endl;
    //std::unordered_map<std::string, torch::Tensor> mapping;

    /*auto leftVectors_t = left_vectors.t();
    auto rightVectors_t = right_vectors.t();*/

    for (int i = 0; i < dict().size(); i++) {
        std::string word = dict()[i];

        //mapping.emplace(word, leftVectors_t[i] + rightVectors_t[i]);
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
