//
// Created by vlad on 17.04.19.
//

#include <iostream>
#include <glove/GloVeBuilder.h>
#include <base/EmbeddingImpl.h>

int main() {
    std::string dict_path("hobbit.txt");
    std::unique_ptr<Embedding<std::string>::Builder> glove(new GloVeBuilder(dict_path));
    std::unique_ptr<Embedding<std::string>> impl = glove->build();

    std::vector<std::string> exceptions = {"the", "a", "and", "of", "to", "but",
                                      "so", "as", "now","him"};
    std::string w;

    /*while (true) {
        std::cout << "Enter word " << std::endl;
        std::cin >> w;
        auto words = glove->closest_words_except(w, 7, exceptions);
        for (auto entry : words) {
            std::cout << entry << std::endl;
        }
    }*/
    //std::cout << dict_path << std::endl;
    return 0;
}