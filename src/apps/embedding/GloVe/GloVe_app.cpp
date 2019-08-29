//
// Created by vlad on 17.04.19.
//

#include <iostream>
#include <glove/GloVeBuilder.h>

int main() {
    std::string dict_path("hobbit.txt");
    GloVeBuilder test(dict_path);
    test.fit();
    std::cout << dict_path << std::endl;
    return 0;
}