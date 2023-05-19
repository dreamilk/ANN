#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include "layer.hpp"
#include "input_layer.hpp"
#include "output_layer.hpp"
#include "hidden_layer.hpp"

#include <vector>

class network
{
private:
    std::vector<layer*> layers;

    double calculateLoss(std::vector<double> output);
public:
    network();
    network(std::vector<int> layerSize);
    ~network();

    void train(std::vector<std::vector<double>> x,std::vector<std::vector<double>> y,int epoches);
    std::vector<double> predict(std::vector<double> input);
    void fprop(std::vector<double> input);
    void bprop(double loss);

};

#endif