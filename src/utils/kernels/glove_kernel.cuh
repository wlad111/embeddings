#ifndef GLOVE_KERNEL_CUH
#define GLOVE_KERNEL_CUH


#include <cuda_runtime>
#include <cstdint>
#include <cmath>


    struct cooc_token {
        float cooccurence;
        size_t i;
        size_t j;
    };


    struct long_vector_array {
        long* elements;
        size_t size;
    };

    struct float_vector_array {
        float* elements;
        size_t  size;
    };

    struct matrix {
        float* elements;
        size_t num_rows;
        size_t num_cols;
    };

    __device__ long get_vector_el(long_vector_array &v, size_t i){
        if (i < v.size) {
            return v.elements[i];
        }
        return cudaErrorInvalidDevicePointer;
    }

    __device__ float get_matrix_el(matrix &m, size_t row, size_t col){
        if (row < m.num_rows && col < m.num_cols) {
            return m.elements[row][col];
        }
        return cudaErrorInvalidDevicePointer;
    }

    __device__ long get_sparse_cooc_el(sparse_cooc_raw &cooc, size_t row, size_t col) {
        if (row < cooc.num_rows && col < cooc.row_lengths[row]){
            return cooc.elements[row][col];
        }
        return cudaErrorInvalidDevicePointer;
    }

    __device__ long_vector_array get_cooc_row(sparse_cooc_raw &cooc, size_t row) {
    long_vector_array c_row;
    c_row.size = cooc.row_lengths[row];
    c_row.elements = cooc.elements[row];
}

__device__ float weighting_func(float x) {
    return x < 10 ? powf(x / 10, 0.75) : 1;
}

    // fit kernel for GLoVe
    __global__ void fit_kernel(sparse_cooc_raw &cooc,
                               matrix &left_vectors,
                               matrix &right_vectors,
                               float_vector_array &bias_left,
                               float_vector_array &bias_right,
                               matrix &softmax_left,
                               matrix &softmax_right,
                               float_vector_array &soft_bias_left,
                               float_vector_array &soft_bias_right,
                               float_vector_array &dL,
                               float_vector_array &dR,
                               int dim
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < cooc.num_rows) {
            long_vector_array cooc_row = get_cooc_row(cooc, i);
            for (size_t packed = 0; packed < cooc_row.size; packed++) {
                int j = packed >> 32;
                int32_t int_to_float = (int32_t)(packed & 0xFFFFFFFFL);
                float X_ij = *(float*)(&int_to_float);
                float asum = 0;
                for (int k = 0; k < dim; k++) {
                    asum += left_vectors.elements[i][k]*right_vectors.elements[j][k];
                }
                float diff = bias_left.elements[i] + bias_right.elements[j] + asum - logf(X_ij);
                float weight = weighting_func(X_ij);
                float fdiff = 0.01 * diff * weight;

                for (int id = 0; id < dim; id++) {
                    float dL = fdiff * right_vectors.elements[j][id];
                    float dR = fdiff * left_vectors.elements[i][id];
                    left_vectors.elements[i][id] -= dL / sqrtf(softmax_left.elements[i][id]);
                    right_vectors.elements[j][id] -= dR / sqrtf(softmax_right.elements[j][id]);
                    softmax_left.elements[i][id] += dL * dL;
                    softmax_right.elements[j][id] += dR * dR;
                }
                bias_left.elements[i] -= fdiff / sqrtf(soft_bias_left.elements[i]);
                bias_right.elements[j] -= fdiff / sqrt(soft_bias_right.elements[j]);
                soft_bias_left.elements[i] += fdiff * fdiff;
                soft_bias_right.elements[j] += fdiff *fdiff;
            }
        }
    }


#endif