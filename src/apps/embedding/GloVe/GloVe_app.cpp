//
// Created by vlad on 17.04.19.
//

#include <iostream>
#include <base/CoocBasedBuilder.h>

int main() {
    std::string dict_path("text8");
    CoocBasedBuilder test(dict_path);
    test.fit();
    std::cout << dict_path << std::endl;
    return 0;
}