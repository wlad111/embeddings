project(embeddings_src_utils_kernels)

add_library(kernels
        glove_kernel.cuh
        glove_kernel.cu)