project(embeddings_src_embedding_glove)

add_library(glove
        GloVeBuilder.h
        GloVeBuilder.cpp)

target_link_libraries(glove core util base vec_tools)

target_include_directories(glove INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
        )