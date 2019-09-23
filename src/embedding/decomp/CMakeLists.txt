project(embeddings_src_embedding_decomp)

add_library(decomp
        DecompBuilder.h
        DecompBuilder.cpp
        MultiDecompBuilder.h
        MultiDecompBuilder.cpp)

target_link_libraries(decomp base)

target_include_directories(decomp INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
        )