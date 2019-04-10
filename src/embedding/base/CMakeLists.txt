project(embeddings_src_embedding_base)

add_library(base
        Embedding.h
        Embedding.cpp
        EmbeddingImpl.cpp
        EmbeddingImpl.h EmbeddingBuilderBase.cpp EmbeddingBuilderBase.h)

target_link_libraries(base core util)