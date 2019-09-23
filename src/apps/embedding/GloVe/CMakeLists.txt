project(embeddings_src_apps_embedding_glove)

add_executable(glove_app
        GloVe_app.cpp)

add_compile_options(-ltbb)

target_link_libraries(glove_app base decomp glove)

target_include_directories(glove_app INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
        )