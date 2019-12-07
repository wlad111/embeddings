#ifndef GLOVE_KERNEL_CUH
#define GLOVE_KERNEL_CUH


#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 64
#define DIM 64

namespace cuda {

    namespace tools {

        template <typename T>
        __global__ void fill_arr_kernel(T* arr, T value, int64_t size) {
            int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
            while (i < size) {
                arr[i] = value;
                i += gridDim.x * blockDim.x;
            }
        }

        template <typename T>
        void fill_arr(T* arr, T value, int64_t size) {
            if (size > 0) {
                dim3 numBlocks;
                numBlocks.x = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
                numBlocks.y = 1;
                numBlocks.z = 1;
                fill_arr_kernel<T> <<<numBlocks, BLOCK_SIZE>>> (arr, value, size);
            }
        }
    }

    typedef struct {
        int dim;
        float step;
    } options;

    typedef struct {
        int row;
        int col;
        float value;

    } cooc_token_bidim;

    typedef struct {
        int size;
        cooc_token_bidim *tokens;
    } cooc_matrix;

    struct long_vector_array {
        long *elements;
        size_t size;
    };

    struct float_vector_array {
        float *elements;
        size_t size;
    };

    struct matrix {
        float *elements;
        size_t num_rows;
        size_t num_cols;
    };

    /*__device__ float get_matrix_el(matrix &m, size_t row, size_t col){
        if (row < m.num_rows && col < m.num_cols) {
            return m.elements[row][col];
        }
        return cudaErrorInvalidDevicePointer;
    }
*/

    __device__ float weighting_func(float x) {
        return x < 10 ? powf(x / 10, 0.75) : 1;
    }

    // fit kernel for GLoVe
    __global__ void fit_kernel(
            const cooc_matrix cooc,
            matrix left_vectors,
            matrix right_vectors,
            matrix grad_update_left,
            matrix grad_update_right,
            int dim) {
        unsigned int id = threadIdx.x;
        unsigned int token_id = blockIdx.x;
        int i = cooc.tokens[token_id].row;
        int j = cooc.tokens[token_id].col;
        float X_ij = cooc.tokens[token_id].value;

        __shared__ float d_left_vec[BLOCK_SIZE];
        __shared__ float d_right_vec[BLOCK_SIZE];
        __shared__ float d_update_left[BLOCK_SIZE];
        __shared__ float d_update_right[BLOCK_SIZE];
        __shared__ float d_bias_left;
        __shared__ float d_bias_right;
        __shared__ float d_update_bias_left;
        __shared__ float d_update_bias_right;

        //an array to store pairwise products to compute dot product;
        __shared__ float pairwise_product[BLOCK_SIZE];
        __shared__ float diff;
        __shared__ float fdiff;

        d_left_vec[id] = left_vectors.elements[i * BLOCK_SIZE + id];
        d_right_vec[id] = right_vectors.elements[j * BLOCK_SIZE + id];
        d_update_left[id] = grad_update_left.elements[i * BLOCK_SIZE + id];
        d_update_right[id] = grad_update_right.elements[j * BLOCK_SIZE + id];

        __syncthreads();

        pairwise_product[id] = d_left_vec[id] * d_right_vec[id];

        __syncthreads();

        //TODO maybe summarize dot product in multiple threads
        if (threadIdx.x == 0) {
            int sum = 0;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                sum += pairwise_product[i];
            }
            diff = sum + d_bias_left + d_bias_right - logf(X_ij);
        }
        //TODO refactor this
        d_left_vec[id] -= (0.01 * weighting_func(X_ij) * diff * d_right_vec[id]) / sqrtf(d_update_left[id]);
        d_right_vec[id] -= (0.01 * weighting_func(X_ij) * diff * d_left_vec[id]) / sqrtf(d_update_right[id]);
        d_update_left[id] += (0.01 * diff * d_right_vec[id] * weighting_func(X_ij)) *
                             (0.01 * diff * d_right_vec[id] * weighting_func(X_ij));
        d_update_right[id] += (0.01 * diff * d_left_vec[id] * weighting_func(X_ij)) *
                              (0.01 * diff * d_left_vec[id] * weighting_func(X_ij));

        if (threadIdx.x == 0) {
            d_bias_left -= 0.01 * diff * weighting_func(X_ij) / sqrtf(d_update_bias_left);
            d_bias_right -= 0.01 * diff * weighting_func(X_ij) / sqrtf(d_update_bias_right);
            d_update_bias_left += 0.01 * diff * weighting_func(X_ij);
            d_update_bias_right += 0.01 * diff * weighting_func(X_ij);
        }

        __syncthreads();
        left_vectors.elements[i * BLOCK_SIZE + id] = d_left_vec[id];
        right_vectors.elements[j * BLOCK_SIZE + id] = d_right_vec[id];
        grad_update_left.elements[i * BLOCK_SIZE + id] = d_update_left[id];
        grad_update_right.elements[j * BLOCK_SIZE + id] = d_update_right[id];
    }

    void fit(cooc_matrix* coocMatrix) {
        dim3 numBlocks;

    }

} //namespace cuda

#endif