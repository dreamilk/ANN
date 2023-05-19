#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include "layer.hpp"

#include <vector>

class network
{
private:
    std::vector<layer*> layers;
public:
    network();
    network(std::vector<int> layerSize);
    ~network();

    void train(std::vector<std::vector<double>> x,std::vector<std::vector<double>> y,int epoches);
    std::vector<double> predict(std::vector<double> input);
};

#endif