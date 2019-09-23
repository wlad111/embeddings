//
// Created by vlad on 17.04.19.
//

#include <iostream>
#include <glove/GloVeBuilder.h>

int main() {
    std::string dict_path("text8");
    GloVeBuilder test(dict_path);
    test.fit();
    //test.write_mapping("mapping.txt");

    std::vector<std::string> exceptions = {"the", "a", "and", "of", "to", "but",
                                      "so", "as", "now","him"};
    std::string w;

    /*while (true) {
        std::cout << "Enter word " << std::endl;
        std::cin >> w;
        auto words = test.closest_words_except(w, 7, exceptions);
        for (auto entry : words) {
            std::cout << entry << std::endl;
        }*/
    //}
    //std::cout << dict_path << std::endl;
    return 0;
}