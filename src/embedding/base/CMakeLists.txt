project(embeddings_src_embedding_base)

add_library(base
        Embedding.h
        Embedding.cpp
        EmbeddingImpl.cpp
        EmbeddingImpl.h EmbeddingBuilderBase.cpp EmbeddingBuilderBase.h CoocBasedBuilder.cpp CoocBasedBuilder.h)

target_link_libraries(base "${TBB_IMPORTED_TARGETS}" "${TORCH_LIBRARIES}")

