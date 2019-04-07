project(embeddings_src_embedding_base)

add_library(base Embedding.h)
target_link_libraries(base core util)