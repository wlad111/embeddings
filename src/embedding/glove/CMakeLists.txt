project(embeddings_src_embedding_glove)

add_library(glove
        GloVeBuilder.h
        GloVeBuilder.cpp)

target_link_libraries(glove core util)