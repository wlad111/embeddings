//
// Created by vlad on 07.04.19.
//

#ifndef EMBEDDINGS_GLOVEBUILDER_H
#define EMBEDDINGS_GLOVEBUILDER_H


#include <base/CoocBasedBuilder.h>
#include <random>



class GloVeBuilder :
        public CoocBasedBuilder {
public:
    void alpha(double alpha);

    GloVeBuilder(std::string &dictPath);

    void x_max(double x_max);
    void dim(int dim);

    void fit () override;

protected:
private:
    double x_max_ = 10;
    double alpha_ = 0.75;
    int dim_ = 50;

    double weightingFunc(double x);

    double initializeValue();
    std::random_device rd;
    std::mt19937 mt; //mersenne twister engine
    std::uniform_real_distribution<double> dist;

};


#endif //EMBEDDINGS_GLOVEBUILDER_H
