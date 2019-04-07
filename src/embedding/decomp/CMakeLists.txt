project(embeddings_src_embedding_decomp)

add_library(decomp
        DecompBuilder.h
        DecompBuilder.cpp
        MultiDecompBuilder.h
        MultiDecompBuilder.cpp)

target_link_libraries(decomp core util)